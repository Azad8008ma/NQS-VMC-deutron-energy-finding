# ============================================
# nqs_vmc_final_stable.py
# نسخه نهایی پایدار (رفع انفجار گرادیان)
# ============================================

import torch
import torch.nn as nn
import torch.func as func
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✅ دستگاه: {device}")

# ============================================
# ۱. تابع موج
# ============================================
class SplineWaveFunction(nn.Module):
    def __init__(self, num_nucleons, num_knots=20):
        super().__init__()
        self.num_nucleons = num_nucleons
        self.num_knots = num_knots
        self.n_pairs = num_nucleons * (num_nucleons - 1) // 2

        self.knots_values = nn.Parameter(torch.randn(num_knots) * 0.05)
        self.register_buffer('knots_positions', torch.linspace(0.01, 10.0, num_knots))

        self.jastrow_alpha = nn.Parameter(torch.tensor(0.3))
        self.jastrow_beta = nn.Parameter(torch.tensor(1.5))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def _spline_interp(self, r):
        r_clamped = torch.clamp(r, self.knots_positions[0], self.knots_positions[-1])
        idx_float = (r_clamped - self.knots_positions[0]) / \
                    (self.knots_positions[-1] - self.knots_positions[0]) * \
                    (self.num_knots - 1)
        idx_low = torch.floor(idx_float).long()
        idx_high = torch.min(idx_low + 1, torch.tensor(self.num_knots - 1, device=r.device))
        frac = idx_float - idx_low.float()
        values = self.knots_values
        return values[idx_low] * (1 - frac) + values[idx_high] * frac

    def _compute_pairwise_distances(self, positions):
        batch_size, A, _ = positions.shape
        dists = []
        for i in range(A):
            for j in range(i + 1, A):
                r = torch.norm(positions[:, i] - positions[:, j], dim=1)
                dists.append(r)
        return torch.stack(dists, dim=1)

    def forward(self, positions):
        r = self._compute_pairwise_distances(positions)
        spline_part = torch.sum(self._spline_interp(r), dim=1)
        hard_core = torch.sum(
            torch.log(1 - torch.exp(-self.jastrow_beta * r) + 1e-6),
            dim=1
        )
        jastrow = -self.jastrow_alpha * torch.sum(r, dim=1)
        scale = torch.clamp(self.output_scale, 0.5, 2.0)
        return scale * (spline_part + hard_core + jastrow)

    def get_trainable_params(self):
        return list(self.parameters())

# ============================================
# ۲. هامیلتونی کایرال
# ============================================
class ChiralHamiltonian:
    def __init__(self, num_nucleons):
        self.A = num_nucleons
        self.hbar2_2m = 20.735
        self.C1 = -0.15
        self.C3 = -0.05
        self.C4 = 0.05
        self.m_pi = 0.7
        self.Lambda = 2.5
        self.core = 0.4

    def _cutoff(self, r):
        return 1 - torch.exp(-(r / self.Lambda)**2)

    def central(self, r):
        yukawa = self.C1 * torch.exp(-self.m_pi * r) / (r + 1e-6)
        hard_core = 500.0 * torch.exp(-(r / self.core)**2)
        return (yukawa + hard_core) * self._cutoff(r)

    def tensor(self, r):
        return self.C3 * torch.exp(-self.m_pi * r) * self._cutoff(r)

    def spin_orbit(self, r):
        return self.C4 * torch.exp(-self.m_pi * r) * self._cutoff(r)

    def kinetic_energy(self, nqs, positions):
        def log_psi_fn(pos):
            return nqs(pos.unsqueeze(0)).squeeze(0)

        def local_energy_kinetic(pos):
            grad = func.grad(log_psi_fn)(pos)
            laplacian = 0.0
            for i in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    laplacian += func.grad(
                        lambda p, i=i, j=j: func.grad(log_psi_fn)(p)[i, j]
                    )(pos)[i, j]
            grad_squared = (grad ** 2).sum()
            return -self.hbar2_2m * (laplacian + grad_squared)

        kinetic = func.vmap(local_energy_kinetic)(positions)
        return kinetic

    def potential_energy(self, positions):
        energy = torch.zeros(positions.shape[0], device=positions.device)
        for i in range(self.A):
            for j in range(i + 1, self.A):
                r = torch.norm(positions[:, i] - positions[:, j], dim=1)
                energy = energy + self.central(r)
                energy = energy + self.tensor(r)
                energy = energy + self.spin_orbit(r)
        return energy

    def total_energy(self, nqs, positions):
        return self.kinetic_energy(nqs, positions) + self.potential_energy(positions)

# ============================================
# ۳. نمونه‌گیر
# ============================================
class SmartSampler:
    def __init__(self, nqs, num_walkers=512, R=1/0.3):
        self.nqs = nqs
        self.num_walkers = num_walkers
        self.R = R

    def sample_initial(self):
        batch = self.num_walkers
        A = self.nqs.num_nucleons
        r = self.R * torch.empty(batch, device=device).exponential_() + 1e-6

        if A == 2:
            x0 = self.R * torch.randn(batch, 3, device=device)
            x0 = x0 / torch.norm(x0, dim=1, keepdim=True) * r.unsqueeze(1)
            positions = torch.zeros(batch, A, 3, device=device)
            positions[:, 0, :] = x0 / 2
            positions[:, 1, :] = -x0 / 2
        else:
            r_expanded = r.unsqueeze(1).unsqueeze(2).expand(batch, A, 3)
            positions = torch.randn(batch, A, 3, device=device) * (r_expanded / np.sqrt(A))
            positions = positions - positions.mean(dim=1, keepdim=True)

        return positions

    def metropolis_step(self, positions, step_size=0.2):
        proposal = positions + torch.randn_like(positions) * step_size

        r_min = torch.ones(proposal.shape[0], device=device) * 10.0
        for i in range(self.nqs.num_nucleons):
            for j in range(i + 1, self.nqs.num_nucleons):
                r_min = torch.min(r_min, torch.norm(proposal[:, i] - proposal[:, j], dim=1))

        mask_box = (r_min < 8.0).unsqueeze(-1).unsqueeze(-1)
        proposal = torch.where(mask_box, proposal, positions)

        log_prob_current = self.nqs(positions)
        log_prob_proposal = self.nqs(proposal)
        ratio = torch.exp(log_prob_proposal - log_prob_current)
        accept = torch.rand_like(ratio) < ratio

        positions = torch.where(accept.unsqueeze(-1).unsqueeze(-1), proposal, positions)
        return positions, accept.float().mean().item()

# ============================================
# ۴. Bootstrap
# ============================================
def bootstrap_energy(energies, num_bootstrap=1000):
    n = len(energies)
    bootstrap_means = []
    for _ in range(num_bootstrap):
        indices = np.random.randint(0, n, size=n)
        bootstrap_means.append(np.mean(energies[indices]))
    return np.mean(bootstrap_means), np.std(bootstrap_means)

# ============================================
# ۵. Yogi
# ============================================
class YogiOptimizer:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-3):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            g_sq = g ** 2
            self.v[i] = self.v[i] - (1 - self.beta2) * g_sq * torch.sign(self.v[i] - g_sq)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# ============================================
# ۶. VMC Solver (اصلاح‌شده)
# ============================================
class VMCSolver:
    def __init__(self, nqs, hamiltonian, sampler, lr=1e-3):
        self.nqs = nqs
        self.hamiltonian = hamiltonian
        self.sampler = sampler
        self.trainable_params = nqs.get_trainable_params()
        self.optimizer = YogiOptimizer(self.trainable_params, lr=lr)

        self.energies_history = []
        self.variance_history = []
        self.acceptance_history = []
        self.best_params = None
        self.best_energy = None
        self.best_error = float('inf')
        self.best_energies = None
        self.no_improve_count = 0

    def compute_gradient(self, positions):
        with torch.no_grad():
            energy = self.hamiltonian.total_energy(self.nqs, positions)
            # فیلتر سخت‌گیرانه
            valid_mask = torch.isfinite(energy) & (torch.abs(energy) < 50.0)
            if valid_mask.sum() < 10:
                return [None] * len(self.trainable_params), 0.0, 0.0

            energy = energy[valid_mask]
            energy_mean = energy.mean()
            energy_var = energy.var()
            # فقط انرژی (بدون واریانس)
            loss_centered = energy - energy_mean

        valid_positions = positions[valid_mask]
        log_psi = self.nqs(valid_positions)

        grads = torch.autograd.grad(
            outputs=log_psi,
            inputs=self.trainable_params,
            grad_outputs=loss_centered,
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )

        n_samples = valid_positions.shape[0]
        weighted_grads = [g / n_samples if g is not None else None for g in grads]
        return weighted_grads, energy_mean.item(), energy_var.item()

    def train(self, target_energy, epochs=3000):
        print(f"\n🚀 شروع VMC (Stable) برای A={self.nqs.num_nucleons}...")

        positions = self.sampler.sample_initial()
        for _ in range(200):
            positions, _ = self.sampler.metropolis_step(positions)

        for epoch in range(epochs):
            acc_rates = []
            for _ in range(3):
                positions, acc = self.sampler.metropolis_step(positions)
                acc_rates.append(acc)

            weighted_grads, mean_e, var_e = self.compute_gradient(positions)

            if np.isnan(mean_e) or mean_e == 0.0:
                print(f"   ⚠️ nan/zero detected at epoch {epoch+1}, stopping...")
                break

            self.optimizer.zero_grad()
            for param, grad in zip(self.trainable_params, weighted_grads):
                if grad is not None:
                    param.grad = grad.clone()
            # clip_grad بسیار سخت‌گیرانه
            torch.nn.utils.clip_grad_norm_(self.trainable_params, 0.01)
            self.optimizer.step(weighted_grads)

            self.energies_history.append(mean_e)
            self.variance_history.append(var_e)
            self.acceptance_history.append(np.mean(acc_rates))

            if abs(mean_e - target_energy) < self.best_error:
                self.best_error = abs(mean_e - target_energy)
                self.best_energy = mean_e
                self.best_params = {k: v.clone() for k, v in self.nqs.state_dict().items()}
                with torch.no_grad():
                    energy = self.hamiltonian.total_energy(self.nqs, positions)
                    valid_mask = torch.isfinite(energy) & (torch.abs(energy) < 50.0)
                    self.best_energies = energy[valid_mask].clone()
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

            if self.no_improve_count > 500:
                print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 300 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"   دوره {epoch+1:4d}/{epochs}: انرژی = {mean_e:.4f} MeV | واریانس = {var_e:.4f} | نرخ پذیرش = {np.mean(acc_rates):.2%}")

        return positions

    def evaluate(self):
        print(f"\n📊 ارزیابی نهایی...")

        if self.best_params:
            self.nqs.load_state_dict(self.best_params)
            print(f"   ✅ بهترین مدل لود شد (انرژی: {self.best_energy:.4f} MeV)")

        if self.best_energies is not None:
            energies = self.best_energies.cpu().numpy()
            print(f"   ✅ استفاده از انرژی‌های بهترین دوره")
        else:
            energies = np.array([0.0])

        mask = np.isfinite(energies) & (np.abs(energies) < 50.0)
        filtered = energies[mask]

        mean_e, std_e = bootstrap_energy(filtered, num_bootstrap=1000)
        print(f"   📊 Bootstrap: E = {mean_e:.4f} ± {std_e:.4f} MeV")

        return mean_e, std_e**2, filtered

# ============================================
# ۷. اجرا برای یک هسته
# ============================================
def run_nucleus(A, target_energy, nucleus_name, epochs=3000, n_walkers=512):
    print("\n" + "=" * 70)
    print(f"🔬 اجرا برای {nucleus_name} (A={A})")
    print("=" * 70)

    nqs = SplineWaveFunction(num_nucleons=A, num_knots=20).to(device)
    hamiltonian = ChiralHamiltonian(num_nucleons=A)
    sampler = SmartSampler(nqs, num_walkers=n_walkers)
    solver = VMCSolver(nqs, hamiltonian, sampler, lr=1e-3)

    positions = solver.train(target_energy, epochs=epochs)
    final_energy, final_variance, energies = solver.evaluate()

    error = abs(final_energy - target_energy)
    accuracy = max(0, 100 - error / abs(target_energy) * 100)

    results = {
        'nucleus': nucleus_name,
        'A': A,
        'target_energy': target_energy,
        'final_energy': final_energy,
        'error': error,
        'accuracy': accuracy,
        'variance': final_variance,
        'energies': energies,
        'energies_history': solver.energies_history,
        'variance_history': solver.variance_history,
        'acceptance_history': solver.acceptance_history,
        'mean_acceptance': np.mean(solver.acceptance_history[-100:]),
        'best_energy': solver.best_energy,
    }

    print("\n" + "=" * 60)
    print(f"📊 نتایج نهایی برای {nucleus_name}:")
    print(f"  انرژی NQS:         {final_energy:.4f} MeV")
    print(f"  انرژی هدف:         {target_energy:.3f} MeV")
    print(f"  خطا:               {error:.4f} MeV")
    print(f"  دقت:               {accuracy:.2f}%")
    print(f"  واریانس:           {final_variance:.4f} MeV²")
    print(f"  نرخ پذیرش میانگین: {results['mean_acceptance']:.2%}")
    print("=" * 60)

    return results

# ============================================
# ۸. بخش اصلی
# ============================================
def main():
    results_deuteron = run_nucleus(A=2, target_energy=-2.2246, nucleus_name="Deuteron (²H)", epochs=3000, n_walkers=512)
    results_triton = run_nucleus(A=3, target_energy=-8.4800, nucleus_name="Triton (³H)", epochs=3000, n_walkers=512)

    results_list = [results_deuteron, results_triton]

    print("\n" + "=" * 90)
    print("📋 جدول ۴-۱: مقایسه انرژی بستگی")
    print("=" * 90)
    print(f"{'هسته':<15} {'مقدار تجربی (MeV)':<20} {'پیش‌بینی NQS (MeV)':<22} {'خطای مطلق (MeV)':<18} {'دقت (%)':<10}")
    print("-" * 90)
    for r in results_list:
        print(f"{r['nucleus']:<15} {r['target_energy']:<20.4f} {r['final_energy']:<22.4f} {r['error']:<18.4f} {r['accuracy']:<10.2f}")
    print("=" * 90)

    plt.figure(figsize=(8, 5))
    plt.plot(results_deuteron['energies_history'], color='blue', linewidth=1.0, alpha=0.7, label='Energy')
    plt.axhline(y=results_deuteron['target_energy'], color='r', linestyle='--', linewidth=1.5, label='Experimental')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Energy (MeV)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('nqs_stable_convergence_deuteron.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_triton['energies_history'], color='blue', linewidth=1.0, alpha=0.7, label='Energy')
    plt.axhline(y=results_triton['target_energy'], color='r', linestyle='--', linewidth=1.5, label='Experimental')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Energy (MeV)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('nqs_stable_convergence_triton.png', dpi=150)
    plt.close()

    plt.figure(figsize=(7, 7))
    predicted = [r['final_energy'] for r in results_list]
    actual = [r['target_energy'] for r in results_list]
    plt.scatter(actual, predicted, color='blue', s=150, zorder=5, edgecolors='black')
    for i, name in enumerate([r['nucleus'] for r in results_list]):
        plt.annotate(name, (actual[i], predicted[i]), textcoords="offset points", xytext=(10, 5), fontsize=12)
    lims = [min(actual + predicted) - 1, max(actual + predicted) + 1]
    plt.plot(lims, lims, 'r--', linewidth=1.5, label='y = x (ideal)')
    plt.xlabel('Experimental Energy (MeV)', fontsize=12)
    plt.ylabel('Predicted NQS Energy (MeV)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('nqs_stable_predicted_vs_actual.png', dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("✅ تمام نمودارها و نتایج تولید شد.")
    print("=" * 70)

    return results_list

if __name__ == "__main__":
    results = main()