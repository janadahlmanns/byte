import numpy as np
import matplotlib.pyplot as plt

def run_iterations(update_fn, w_init, eta, input_val, num_iters=100):
    """Run weight update for num_iters iterations."""
    w = w_init
    weights = [w]
    for _ in range(num_iters):
        w = update_fn(w, eta, input_val)
        # Clamp to [-1, 1] for display
        w = np.clip(w, -1, 1)
        weights.append(w)
    return weights

# Define update functions
def fn_A(w, eta, inp):
    """A) w_new = w_old * (1 + eta*input*(1-|w_old|)*|w_old|)"""
    return w * (1 + eta * inp * (1 - abs(w)) * abs(w))

def fn_B(w, eta, inp):
    """B) w_new = sign(w_old) * (|w_old| + eta*input) * (1-|w_old|)"""
    sign = np.sign(w) if w != 0 else 1
    return sign * (abs(w) + eta * inp) * (1 - abs(w))

def fn_C(w, eta, inp):
    """C) w_new = w_old + eta*input * (1 - w_old^2)"""
    return w + eta * inp * (1 - w**2)

def fn_D(w, eta, inp):
    """D) w_new = w_old * (1 + eta*input) / (1 + |eta*input|)"""
    denominator = 1 + abs(eta * inp)
    if denominator == 0:
        return w
    return w * (1 + eta * inp) / denominator

def fn_E(w, eta, inp):
    """E) w_new = tanh(w_old + eta*input*sign(w_old)*(1 - |w_old|))"""
    sign = np.sign(w) if w != 0 else 1
    arg = w + eta * inp * sign * (1 - abs(w))
    return np.tanh(arg)

def fn_F(w, eta, inp):
    """F) w_new = tanh(w_old + eta*input)  [simple accumulator]"""
    return np.tanh(w + eta * inp)

def fn_G(w, eta, inp):
    """G) w_new = -sin(π*w_old)"""
    return -np.sin(np.pi * w)

def fn_H(w, eta, inp):
    """H) w_new = 1/π * cos(π*w_old)"""
    return (1 / np.pi) * np.cos(np.pi * w)

def fn_I(w, eta, inp):
    """I) w_new = -π * cos(π*w_old)"""
    return -np.pi * np.cos(np.pi * w)

def fn_J(w, eta, inp):
    """J) w_new = (w_old)^3"""
    return w**3

def fn_K(w, eta, inp):
    """K) w_new = sign(w_old) * max(0, |w_old| + η*|w_old|*(1-|w_old|)*modsum)"""
    sign = np.sign(w) if w != 0 else 1
    abs_w = abs(w)
    magnitude = abs_w + eta * abs_w * (1 - abs_w) * inp
    return sign * max(0, magnitude)

def fn_L(w, eta, inp):
    """L) w_new = clip(w_old * (1 + η*modsum), -1, 1)"""
    return np.clip(w * (1 + eta * inp), -1, 1)

functions = {
    'A': (fn_A, "w_new = w_old * (1 + η·input·(1-|w_old|)·|w_old|)"),
    'B': (fn_B, "w_new = sign(w_old) · (|w_old| + η·input) · (1-|w_old|)"),
    'C': (fn_C, "w_new = w_old + η·input · (1 - w_old²)"),
    'D': (fn_D, "w_new = w_old · (1 + η·input) / (1 + |η·input|)"),
    'E': (fn_E, "w_new = tanh(w_old + η·input·sign(w_old)·(1 - |w_old|))"),
    'F': (fn_F, "w_new = tanh(w_old + η·input)"),
    'G': (fn_G, "w_new = -sin(π·w_old)"),
    'H': (fn_H, "w_new = 1/π · cos(π·w_old)"),
    'I': (fn_I, "w_new = -π · cos(π·w_old)"),
    'J': (fn_J, "w_new = (w_old)³"),
    'K': (fn_K, "w_new = sign(w)·max(0, |w|+η|w|(1-|w|)·modsum)"),
    'L': (fn_L, "w_new = clip(w(1+η·modsum), -1, 1)"),
}

param_sets = [
    (0.1, 1, "eta=0.1, input=1"),
    (0.1, -1, "eta=0.1, input=-1"),
    (0.2, 3, "eta=0.2, input=3"),
]

w_init = 0.1

# Generate plots for each function
for fn_name, (fn, formula) in functions.items():
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Function {fn_name}: {formula}', fontsize=12, fontweight='bold')
    
    # Top row: w_init = 0.1
    for idx, (eta, inp, label) in enumerate(param_sets):
        weights = run_iterations(fn, 0.1, eta, inp, num_iters=100)
        
        ax = axes[0, idx]
        ax.plot(weights, linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Weight')
        ax.set_title(label + " (w_init=+0.1)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='w=1.0')
        ax.axhline(y=-1.0, color='r', linestyle='--', alpha=0.3, label='w=-1.0')
        ax.set_ylim([-1.2, 1.2])
        ax.legend(fontsize=8)
        
        # Print convergence point
        final_w = weights[-1]
        print(f"Function {fn_name}, {label} (w_init=+0.1): converges to w ≈ {final_w:.6f}")
    
    # Middle row: w_init = 0.0
    for idx, (eta, inp, label) in enumerate(param_sets):
        weights = run_iterations(fn, 0.0, eta, inp, num_iters=100)
        
        ax = axes[1, idx]
        ax.plot(weights, linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Weight')
        ax.set_title(label + " (w_init=0.0)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='w=1.0')
        ax.axhline(y=-1.0, color='r', linestyle='--', alpha=0.3, label='w=-1.0')
        ax.set_ylim([-1.2, 1.2])
        ax.legend(fontsize=8)
        
        # Print convergence point
        final_w = weights[-1]
        print(f"Function {fn_name}, {label} (w_init=0.0): converges to w ≈ {final_w:.6f}")
    
    # Bottom row: w_init = -0.1
    for idx, (eta, inp, label) in enumerate(param_sets):
        weights = run_iterations(fn, -0.1, eta, inp, num_iters=100)
        
        ax = axes[2, idx]
        ax.plot(weights, linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Weight')
        ax.set_title(label + " (w_init=-0.1)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='w=1.0')
        ax.axhline(y=-1.0, color='r', linestyle='--', alpha=0.3, label='w=-1.0')
        ax.set_ylim([-1.2, 1.2])
        ax.legend(fontsize=8)
        
        # Print convergence point
        final_w = weights[-1]
        print(f"Function {fn_name}, {label} (w_init=-0.1): converges to w ≈ {final_w:.6f}")
    
    plt.tight_layout()
    plt.savefig(f'weight_scaling/plasticity_fn_{fn_name}.png', dpi=150)
    print(f"Saved: plasticity_fn_{fn_name}.png\n")
    plt.close()

# Create combined plot (eta=0.1, input=1 only)
eta, inp = 0.1, 1
fig, ax = plt.subplots(figsize=(14, 6))

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'cyan', 'gray', 'olive', 'navy', 'teal']
for (fn_name, (fn, formula)), color in zip(functions.items(), colors):
    weights = run_iterations(fn, w_init, eta, inp, num_iters=100)
    ax.plot(weights, linewidth=2.5, marker='o', markersize=4, label=f'{fn_name}: {formula}', color=color, alpha=0.8)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Weight', fontsize=12)
ax.set_title('All Functions Compared: η=0.1, input=1 (100 iterations)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=-1.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=0.0, color='k', linestyle='-', alpha=0.1, linewidth=0.5)
ax.set_ylim([-1.2, 1.2])
ax.legend(fontsize=9, loc='best')
plt.tight_layout()
plt.savefig('weight_scaling/plasticity_all_functions.png', dpi=150)
print("Saved: plasticity_all_functions.png")
plt.close()

# Create combined plot (eta=0.2, input=-1 only)
eta, inp = 0.2, -1
fig, ax = plt.subplots(figsize=(14, 6))

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'cyan', 'gray', 'olive', 'navy', 'teal']
for (fn_name, (fn, formula)), color in zip(functions.items(), colors):
    weights = run_iterations(fn, w_init, eta, inp, num_iters=100)
    ax.plot(weights, linewidth=2.5, marker='o', markersize=4, label=f'{fn_name}: {formula}', color=color, alpha=0.8)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Weight', fontsize=12)
ax.set_title('All Functions Compared: η=0.2, input=-1 (100 iterations)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=-1.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=0.0, color='k', linestyle='-', alpha=0.1, linewidth=0.5)
ax.set_ylim([-1.2, 1.2])
ax.legend(fontsize=9, loc='best')
plt.tight_layout()
plt.savefig('weight_scaling/plasticity_all_functions_eta02_input-1.png', dpi=150)
print("Saved: plasticity_all_functions_eta02_input-1.png")
plt.close()

# Create combined plot (eta=0.5, input=-1 only)
eta, inp = 0.5, -1
fig, ax = plt.subplots(figsize=(14, 6))

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'cyan', 'gray', 'olive', 'navy', 'teal']
for (fn_name, (fn, formula)), color in zip(functions.items(), colors):
    weights = run_iterations(fn, w_init, eta, inp, num_iters=100)
    ax.plot(weights, linewidth=2.5, marker='o', markersize=4, label=f'{fn_name}: {formula}', color=color, alpha=0.8)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Weight', fontsize=12)
ax.set_title('All Functions Compared: η=0.5, input=-1 (100 iterations)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=-1.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=0.0, color='k', linestyle='-', alpha=0.1, linewidth=0.5)
ax.set_ylim([-1.2, 1.2])
ax.legend(fontsize=9, loc='best')
plt.tight_layout()
plt.savefig('weight_scaling/plasticity_all_functions_eta05_input-1.png', dpi=150)
print("Saved: plasticity_all_functions_eta05_input-1.png")
plt.close()

# Create combined plot (eta=0.05, input=1 only)
eta, inp = 0.05, 1
fig, ax = plt.subplots(figsize=(14, 6))

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'cyan', 'gray', 'olive', 'navy', 'teal']
for (fn_name, (fn, formula)), color in zip(functions.items(), colors):
    weights = run_iterations(fn, w_init, eta, inp, num_iters=100)
    ax.plot(weights, linewidth=2.5, marker='o', markersize=4, label=f'{fn_name}: {formula}', color=color, alpha=0.8)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Weight', fontsize=12)
ax.set_title('All Functions Compared: η=0.05, input=1 (100 iterations)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=-1.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=0.0, color='k', linestyle='-', alpha=0.1, linewidth=0.5)
ax.set_ylim([-1.2, 1.2])
ax.legend(fontsize=9, loc='best')
plt.tight_layout()
plt.savefig('weight_scaling/plasticity_all_functions_eta0.05_input1.png', dpi=150)
print("Saved: plasticity_all_functions_eta0.05_input1.png")
plt.close()


print("\nAll plots generated successfully!")
