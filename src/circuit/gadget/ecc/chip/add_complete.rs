use super::super::EccInstructions;
use super::{CellValue, EccChip, EccPoint};

use group::Curve;
use halo2::{
    arithmetic::{CurveAffine, Field, FieldExt},
    circuit::{Chip, Region},
    plonk::{ConstraintSystem, Error, Expression},
};

pub(crate) fn create_gate<C: CurveAffine>(
    meta: &mut ConstraintSystem<C::Base>,
    q_add_complete: Expression<C::Base>,
    a: Expression<C::Base>,
    b: Expression<C::Base>,
    c: Expression<C::Base>,
    d: Expression<C::Base>,
    alpha: Expression<C::Base>,
    beta: Expression<C::Base>,
    gamma: Expression<C::Base>,
    delta: Expression<C::Base>,
    lambda: Expression<C::Base>,
    x_p: Expression<C::Base>,
    y_p: Expression<C::Base>,
    x_q: Expression<C::Base>,
    y_q: Expression<C::Base>,
    x_r: Expression<C::Base>,
    y_r: Expression<C::Base>,
) {
    // Boolean checks on A, B, C, D
    {
        meta.create_gate("Check A is boolean", |_| {
            q_add_complete.clone() * a.clone() * (Expression::Constant(C::Base::one()) - a.clone())
        });
        meta.create_gate("Check B is boolean", |_| {
            q_add_complete.clone() * b.clone() * (Expression::Constant(C::Base::one()) - b.clone())
        });
        meta.create_gate("Check C is boolean", |_| {
            q_add_complete.clone() * c.clone() * (Expression::Constant(C::Base::one()) - c.clone())
        });
        meta.create_gate("Check D is boolean", |_| {
            q_add_complete.clone() * d.clone() * (Expression::Constant(C::Base::one()) - d.clone())
        });
    }

    // Logical implications of Boolean flags
    {
        // x_q = x_p ⟹ A
        meta.create_gate("x_q = x_p ⟹ A", |_| {
            let lhs = (x_q.clone() - x_p.clone()) * alpha.clone();
            let rhs = Expression::Constant(C::Base::one()) - a.clone();
            q_add_complete.clone() * (lhs - rhs)
        });

        // x_p = 0 ⟹ B
        meta.create_gate("x_p = 0 ⟹ B", |_| {
            let lhs = x_p.clone() * beta.clone();
            let rhs = Expression::Constant(C::Base::one()) - b.clone();
            q_add_complete.clone() * (lhs - rhs)
        });

        // B ⟹ x_p = 0
        meta.create_gate("B ⟹ x_p = 0", |_| {
            q_add_complete.clone() * b.clone() * x_p.clone()
        });

        // x_q = 0 ⟹ C
        meta.create_gate("x_q = 0 ⟹ C", |_| {
            let lhs = x_q.clone() * gamma.clone();
            let rhs = Expression::Constant(C::Base::one()) - c.clone();
            q_add_complete.clone() * (lhs - rhs)
        });

        // C ⟹ x_q = 0
        meta.create_gate("C ⟹ x_q = 0", |_| {
            q_add_complete.clone() * c.clone() * x_q.clone()
        });

        // y_q = -y_p ⟹ D
        meta.create_gate("y_q = y_p ⟹ D", |_| {
            let lhs = (y_q.clone() + y_p.clone()) * delta.clone();
            let rhs = Expression::Constant(C::Base::one()) - d.clone();
            q_add_complete.clone() * (lhs - rhs)
        });
    }

    // Handle cases in incomplete addition
    {
        // x_q ≠ x_p ⟹ λ = (y_q − y_p)/(x_q − x_p)
        meta.create_gate("x equality", |_| {
            let equal = x_q.clone() - x_p.clone();
            let unequal = equal.clone() * lambda.clone() - (y_q.clone() - y_p.clone());
            q_add_complete.clone() * equal * unequal
        });

        // A ∧ y_p ≠ 0 ⟹ λ = (3 * x_p^2) / 2 * y_p
        meta.create_gate("x equal, y nonzero", |_| {
            let three_x_p_sq =
                Expression::Constant(C::Base::from_u64(3)) * x_p.clone() * x_p.clone();
            let two_y_p = Expression::Constant(C::Base::from_u64(2)) * y_p.clone();
            let gradient = two_y_p * lambda.clone() - three_x_p_sq;
            q_add_complete.clone() * a.clone() * gradient
        });

        // (¬B ∧ ¬C ⟹ x_r = λ^2 − x_p − x_q) ∧ (B ⟹ x_r = x_q)
        meta.create_gate("x_r check", |_| {
            let not_b = Expression::Constant(C::Base::one()) - b.clone();
            let not_c = Expression::Constant(C::Base::one()) - c.clone();
            let x_r_lambda =
                lambda.clone() * lambda.clone() - x_p.clone() - x_q.clone() - x_r.clone();
            let x_r_x_q = b.clone() * (x_r.clone() - x_q.clone());
            q_add_complete.clone() * (not_b * not_c * x_r_lambda - x_r_x_q)
        });

        // ¬B ∧ ¬C ⟹ y_r = λ⋅(x_p − x_r) − y_p) ∧ (B ⟹ y_r = y_q)
        meta.create_gate("y_r check", |_| {
            let not_b = Expression::Constant(C::Base::one()) - b.clone();
            let not_c = Expression::Constant(C::Base::one()) - c.clone();
            let y_r_lambda =
                lambda.clone() * (x_p.clone() - x_r.clone()) - y_p.clone() - y_r.clone();
            let y_r_y_q = b.clone() * (y_r.clone() - y_q.clone());
            q_add_complete.clone() * (not_b * not_c * y_r_lambda - y_r_y_q)
        });

        // C ⟹ x_r = x_p
        meta.create_gate("x_r = x_p when x_q = 0", |_| {
            q_add_complete.clone() * (c.clone() * (x_r.clone() - x_p.clone()))
        });

        // C ⟹ y_r = y_p
        meta.create_gate("y_r = y_p when x_q = 0", |_| {
            q_add_complete.clone() * (c.clone() * (y_r.clone() - y_p.clone()))
        });

        // D ⟹ x_r = 0
        meta.create_gate("D ⟹ x_r = 0", |_| {
            q_add_complete.clone() * d.clone() * x_r.clone()
        });

        // D ⟹ y_r = 0
        meta.create_gate("D ⟹ y_r = 0", |_| {
            q_add_complete.clone() * d.clone() * y_r.clone()
        });
    }
}

pub(super) fn assign_region<C: CurveAffine>(
    a: &<EccChip<C> as EccInstructions<C>>::Point,
    b: &<EccChip<C> as EccInstructions<C>>::Point,
    region: &mut Region<'_, EccChip<C>>,
    config: <EccChip<C> as Chip>::Config,
) -> Result<EccPoint<C::Base>, Error> {
    let (x_p, y_p) = (a.x.value, a.y.value);
    let (x_q, y_q) = (b.x.value, b.y.value);

    config.q_add_complete.enable(region, 0)?;

    // Rename columns here to match specification
    let a = config.add_complete_bool[0];
    let b = config.add_complete_bool[1];
    let c = config.add_complete_bool[2];
    let d = config.add_complete_bool[3];
    let alpha = config.add_complete_inv[0];
    let beta = config.add_complete_inv[1];
    let gamma = config.add_complete_inv[2];
    let delta = config.add_complete_inv[3];

    // Assign A, B, C, D, alpha, beta, gamma, delta
    {
        match (x_p, x_q) {
            (Some(x_p), Some(x_q)) => {
                if x_q == x_p {
                    region.assign_advice(|| "set A", a, 0, || Ok(C::Base::one()))?;
                } else {
                    let alpha_val = (x_q - x_p).invert().unwrap();
                    region.assign_advice(|| "set alpha", alpha, 0, || Ok(alpha_val))?;
                }
                if x_p == C::Base::zero() {
                    region.assign_advice(|| "set B", b, 0, || Ok(C::Base::one()))?;
                } else {
                    let beta_val = x_p.invert().unwrap();
                    region.assign_advice(|| "set beta", beta, 0, || Ok(beta_val))?;
                }
                if x_q == C::Base::zero() {
                    region.assign_advice(|| "set C", c, 0, || Ok(C::Base::one()))?;
                } else {
                    let gamma_val = x_q.invert().unwrap();
                    region.assign_advice(|| "set gamma", gamma, 0, || Ok(gamma_val))?;
                }
            }
            _ => (),
        }

        match (y_p, y_q) {
            (Some(y_p), Some(y_q)) => {
                if y_p == -y_q {
                    region.assign_advice(|| "set D", d, 0, || Ok(C::Base::one()))?;
                } else {
                    let delta_val = (y_q + y_p).invert().unwrap();
                    region.assign_advice(|| "set delta", delta, 0, || Ok(delta_val))?;
                }
            }
            _ => (),
        }
    }

    // Compute R = P + Q
    let r = x_p
        .zip(y_p)
        .zip(x_q)
        .zip(y_q)
        .map(|(((x_p, y_p), x_q), y_q)| {
            let p = C::from_xy(x_p, y_p).unwrap();
            let q = C::from_xy(x_q, y_q).unwrap();
            p + q
        });
    let x_r_val = r.map(|r| r.to_affine().get_xy().unwrap().0);
    let x_r_cell = region.assign_advice(
        || "set x_r",
        config.x_a,
        1,
        || x_r_val.ok_or(Error::SynthesisError),
    )?;
    let y_r_val = r.map(|r| r.to_affine().get_xy().unwrap().1);
    let y_r_cell = region.assign_advice(
        || "set y_r",
        config.y_a,
        1,
        || y_r_val.ok_or(Error::SynthesisError),
    )?;

    Ok(EccPoint {
        x: CellValue::new(x_r_cell, x_r_val),
        y: CellValue::new(y_r_cell, y_r_val),
    })
}
