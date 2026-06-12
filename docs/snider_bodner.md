# Structure-Function Loss: Reference Notes

**Context for Claude Code.** Spec and background for implementing a real-space,
FFT-free structure-function (SF) loss as a spectral-fidelity / cascade-fidelity
term for the SamudraBGC mesoscale emulator (9 km double-gyre, ConvNeXt U-Net,
PyTorch/DDP). Intended as an additive term alongside the pointwise MAE and the
gradient penalty, not a replacement.

**Source.** Snider & Bodner (MIT), "Diagnosing Submesoscale Energy Cascades from
Anisotropic Structure Function Decompositions," Liège 2026 poster. Equations
below were read directly off the rasterized poster (the Cambria Math layer does
not extract cleanly). The poster's application is submesoscale; only the SF
machinery transfers to the mesoscale emulator.

---

## 1. Why an SF loss here

- Real space, no FFT. Works directly on non-periodic, land-masked domains. A
  spectral loss needs periodicity or windowing, which the double-gyre boundaries
  violate. SF is the real-space analog of spectral nudging.
- The third-order SF is signed and odd, so it constrains the direction and
  magnitude of the energy cascade, not just the spectral slope. This targets the
  diagnosed velocity-only failure (excess power below ~200 km wavelength).
- It is a statistical constraint. It pins spectral content and the anisotropy
  budget, not eddy positions. This is consistent with the unpredictable-mesoscale
  / ensemble framing: keep pointwise MAE and the gradient penalty as the primary
  terms.

## 2. Equations (verified from poster)

Longitudinal SF, order n (velocity increment projected on the separation
direction):
```
S^n(r) = < [ ( u(x+r) - u(x) ) . rhat ]^n >
```
`< . >` domain average, `r` separation (scale), `u` velocity, `n` order,
`rhat = r / |r|`.

Third order, idealized turbulence (physics hook, interpretation only):
```
S^3(r) = c0 * eps * r          # linear in scale; eps = dissipation, c0 const
```

Mixed / coupled SF (blends variables v_i; relevant for flow-tracer coupling):
```
< (dv1)^n1 ... (dvm)^nm >
```

SO(2) modal decomposition (angular frequency is 2*m*theta, from the pi-periodicity
of the increment field):
```
S^n(r, theta) = sum_{m=-inf}^{inf} S_m^n(r) * exp( i * (2*m*theta - phi_m^n(r)) )
m  = 0  -> isotropic component
m != 0  -> anisotropic components
```

Mode amplitude (angular transform) and per-mode dissipation:
```
S_m^3(r) = (2/pi) * | integral_0^pi  S^3(r, theta) * exp(-2*i*m*theta) dtheta |
|eps_m|  = S_m^3(r) / (c0 * r)
```

Physical-domain weight map (which cells drive mode m at scale r; diagnostic, not
part of the loss):
```
S_m^3(r) ~ sum_{x,y} sum_{theta in [0,pi)} du^3(x,y; r,theta) * exp(-i*(2*m*theta - phi_m(r)))
         = sum_{x,y} w(x,y)
```

## 3. Key empirical finding

Mode spectra converge fast (poster Fig. 4): the first few m carry essentially all
the information. Truncate the modal target at small |m|. Do not build a
high-dimensional modal loss.

## 4. Discrete forms for implementation

Angle-averaged (m = 0) SF at radial bin r, over valid pairs:
```
S^n(r) = mean over { x, x+d : |d| in bin_r, both cells valid } of
         [ ( field(x+d) - field(x) ) . dhat ]^n      # dot + dhat only for vector fields
```

Modal amplitude from angular bins:
```
S_m^n(r) = | (1 / N_theta) * sum_theta  Shat^n(r, theta) * exp(-2*i*m*theta) |
```
`Shat^n(r, theta)` is the SF estimated within angular bin theta at radius r.

Loss:
```
L_SF = sum_n sum_r  alpha(n, r) * | S^n_pred(r) - S^n_truth(r) |        # robust norm (MAE)
       ( + optional modal terms  sum_{m>=1} beta(m, r) * | S_m^n_pred(r) - S_m^n_truth(r) | )
```
Compute S in (signed) log space when amplitudes span orders of magnitude.
`alpha(n, r)` upweights the scales of interest: the deformation radius and the
bands where the emulator currently misbehaves.

## 5. Implementation spec for this codebase

**Lattice shift bank.** On the uniform 9 km grid, integer offsets `(di, dj)` give
`r = dx * sqrt(di^2 + dj^2)`, `theta = atan2(dj, di)`. Precompute a fixed set up
to a max lag, group by radial bin. Each increment is one shifted subtraction.
Fully differentiable; drops into the existing PyTorch loss.

**Non-periodic / land masking (the main point).** Do NOT use `torch.roll` (it
wraps around the domain). Shift with slicing (no wrap) and build a pair-validity
mask (both endpoints in-ocean). Normalize each radial bin by its valid-pair count.
This is the concrete advantage over an FFT spectral loss on the masked
double-gyre.

**Field handling.**
- Velocity SF needs `(u, v)`. The emulator state is `(psi, phi)`. Reconstruct
  `(u, v)` with the same inverse-Helmholtz the model already uses (rotational from
  psi, divergent from phi), then apply the longitudinal projection. Keep it
  differentiable; do not detach.
- Tracers (DIC, O2, NO3, Chl): use scalar SFs `< [c(x+d) - c(x)]^n >`, no
  projection.
- Optional coupled SF `< (du . dhat) * dc >` encodes flow-tracer covariance at
  scale r (ties to the covariance-propagation goal) but is higher variance. Add
  later, if at all.

**Resolution constraint on anisotropy.** At 9 km the deformation radius
(~30 to 50 km) is only ~3 to 6 cells, so the informative separations are
`r ~ 1 to 10 cells`. Angular resolution at small r is poor (few lattice
directions), so realistically resolve only `m = 0, 1`, maybe 2. The poster
resolves many modes because its grid is 156 m. Do not over-specify the modal
target for this grid; start with `m = 0` only.

**Order choice.**
- `n = 2`: spectral-slope analog (S2 maps to the KE spectrum). Good default, low
  variance.
- `n = 3`: signed, constrains cascade direction and magnitude. Higher variance
  (odd-order cancellation). Use to target the small-scale energy error
  specifically.

**Estimator noise.** A single-snapshot SF is noisy, worse for odd orders. Average
over all valid pairs in the domain and over the batch. Consider light temporal
averaging across rollout steps.

**Cost.** ~O(P * n_shifts), P = pixels, one shifted subtraction + masked reduction
per shift. Cap max lag to the scales of interest; subsample angles at large r
where lattice directions proliferate.

## 6. Reference skeleton (adapt, not drop-in)

```python
def sf_loss(pred, truth, mask, shift_bank, orders=(2, 3), modal_m=(0,),
            vector=False, log_space=True):
    """
    pred, truth : (B, C, H, W). If vector=True, C = (u, v) AFTER inverse-Helmholtz.
    mask        : (B, 1, H, W) or (1, 1, H, W). 1 = valid ocean cell.
    shift_bank  : dict {r_bin: [(di, dj), ...]} of integer lattice offsets.
    Returns a scalar loss. All ops differentiable.
    """
    total = pred.new_zeros(())
    for r_bin, shifts in shift_bank.items():
        for n in orders:
            sp = _sf_estimate(pred,  mask, shifts, n, vector, modal_m)   # tensor over modes
            st = _sf_estimate(truth, mask, shifts, n, vector, modal_m)
            if log_space:
                sp, st = _signed_log(sp), _signed_log(st)   # preserve sign for odd n
            total = total + alpha(n, r_bin) * (sp - st).abs().mean()
    return total


def _masked_increment(f, mask, di, dj, vector):
    # NON-PERIODIC shift via slicing (no wrap); build pair validity from mask.
    a, b, valid = _shift_pair(f, mask, di, dj)            # a = f(x), b = f(x+d)
    inc = b - a
    if vector:                                            # longitudinal projection
        d = torch.tensor([di, dj], dtype=f.dtype, device=f.device)
        dhat = d / d.norm()
        inc = (inc * dhat.view(1, 2, 1, 1)).sum(1, keepdim=True)
    return inc, valid


def _sf_estimate(f, mask, shifts, n, vector, modal_m):
    # m = 0 : pool increments**n over all shifts in the bin and all valid cells
    #         (sum over valid, divide by valid count).
    # modal : bin shifts by theta, take exp(-2 i m theta) weighted sum, then abs().
    ...
```

## 7. Caveats

- `S^3 = c0 * eps * r` is an inertial-range, stationary, isotropic (or 2D/QG
  analog) result. The forced, non-stationary, anisotropic flow will not follow it
  cleanly. The loss does not need it: match the emulator SF to the model SF; the
  c0 / eps reading is for interpreting results only.
- SF is a distributional constraint and does not pin eddy positions by design.

## 8. References to pull

- Xie & Bühler 2018, JFM. doi:10.1017/jfm.2018.528. Exact SF<->spectrum relation;
  wave-vortex decomposition. Most directly useful for turning SF mismatch into a
  spectral-fidelity loss and tying it to `(psi, phi)`.
- Wang & Bühler 2021, JPO. doi:10.1175/JPO-D-20-0199.1. Helmholtz / wave-vortex
  decomposition of SFs. Pairs with the `(psi, phi)` state for separate rotational
  vs divergent targets.
- Biferale & Procaccia 2005, Phys. Rep. doi:10.1016/j.physrep.2005.04.001.
  SO(3)/SO(2) anisotropy machinery the poster builds on.
- Biferale et al. 2002, JFM. doi:10.1017/S0022112001006632. Anisotropy via
  symmetry-group decomposition.
- Pearson et al. 2021, JFM. doi:10.1017/jfm.2021.247. Oceanic SF practice /
  submesoscale anisotropy.
- Poje et al. 2017, Phys. Fluids. doi:10.1063/1.4974331. SF in the ocean
  (dispersion).

---

**Poster model setup (context only; differs from the emulator domain).**
LESStudySetup.jl on Oceananigans.jl; 100 x 100 x 0.25 km domain;
156.25 m x 156.25 m x 1.125 m; doubly periodic; f-plane hydrostatic Boussinesq;
20-day run; forced by fixed mesoscale eddies + surface wind + surface cooling.