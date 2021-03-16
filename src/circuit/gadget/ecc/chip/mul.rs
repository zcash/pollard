use super::super::EccInstructions;
use super::{add_complete, double, CellValue, EccChip, EccPoint};

use ff::PrimeField;
use halo2::{
    arithmetic::{CurveAffine, Field, FieldExt},
    circuit::{Chip, Region},
    plonk::{ConstraintSystem, Error, Expression},
};

pub(super) fn create_gate<C: CurveAffine>(
    meta: &mut ConstraintSystem<C::Base>,
    q_mul: Expression<C::Base>,
    x_a_cur: Expression<C::Base>,
    x_a_next: Expression<C::Base>,
    x_p_cur: Expression<C::Base>,
    x_p_next: Expression<C::Base>,
    lambda1_cur: Expression<C::Base>,
    lambda1_next: Expression<C::Base>,
    lambda2_cur: Expression<C::Base>,
    lambda2_next: Expression<C::Base>,
) {
    let y_a_cur = (lambda1_cur.clone() + lambda2_cur.clone())
        * (x_a_cur.clone()
            - (lambda1_cur.clone() * lambda1_cur.clone() - x_a_cur.clone() - x_p_cur.clone()))
        * C::Base::TWO_INV;

    let y_a_next = (lambda1_next.clone() + lambda2_next)
        * (x_a_next.clone() - (lambda1_next.clone() * lambda1_next - x_a_next.clone() - x_p_next))
        * C::Base::TWO_INV;

    // Double-and-add expr1
    meta.create_gate("Double-and-add expr1", |_| {
        // λ_{2,i}^2 − x_{A,i+1} −(λ_{1,i}^2 − x_{A,i} − x_{P,i}) − x_{A,i} = 0
        let expr1 = lambda2_cur.clone() * lambda2_cur.clone()
            - x_a_next.clone()
            - (lambda1_cur.clone() * lambda1_cur)
            + x_p_cur;

        q_mul.clone() * expr1
    });

    // Double-and-add expr2
    meta.create_gate("Double-and-add expr2", |_| {
        // λ_{2,i}⋅(x_{A,i} − x_{A,i+1}) − y_{A,i} − y_{A,i+1} = 0
        let expr2 = lambda2_cur * (x_a_cur - x_a_next) - y_a_cur - y_a_next;

        q_mul.clone() * expr2
    });
}

pub(super) fn assign_region<C: CurveAffine>(
    scalar: &<EccChip<C> as EccInstructions<C>>::ScalarVar,
    base: &<EccChip<C> as EccInstructions<C>>::Point,
    region: &mut Region<'_, EccChip<C>>,
    config: <EccChip<C> as Chip>::Config,
) -> Result<EccPoint<C::Base>, Error> {
    // Initialise acc := [2] base
    let mut acc = double::assign_region(&base, region, config.clone()).unwrap();
    let mut x_a = acc.x.value.unwrap();
    let mut y_a = acc.y.value.unwrap();
    let mut x_a_cell = region.assign_advice(
        || "x_a",
        config.x_a,
        0,
        || acc.x.value.ok_or(Error::SynthesisError),
    )?;

    assert_eq!(scalar.k_bits.len(), C::Scalar::NUM_BITS as usize);

    // Bits used in incomplete addition. k_{254} to k_{4} inclusive
    let incomplete_range = 0..(C::Scalar::NUM_BITS as usize - 1 - config.num_complete_bits);
    let k_incomplete = &scalar.k_bits[incomplete_range.clone()];

    // Bits used in complete addition. k_{3} to k_{1} inclusive
    let complete_range = (C::Scalar::NUM_BITS as usize - 1 - config.num_complete_bits)
        ..(C::Scalar::NUM_BITS as usize - 1);
    let k_complete = &scalar.k_bits[complete_range.clone()];

    // The least significant bit
    let k_0_row = C::Scalar::NUM_BITS as usize - 1;
    let k_0 = &scalar.k_bits[k_0_row];

    for (row, k) in k_incomplete.iter().enumerate() {
        let x_p = base.x.value.unwrap();
        region.assign_advice(|| "x_p", config.x_p, row, || Ok(x_p))?;

        let mut y_p = base.y.value.unwrap();
        if k.value.unwrap() == C::Base::zero() {
            y_p = -y_p;
        }

        // Compute and assign `lambda1, lambda2`
        let lambda1 = (y_a - y_p) * (x_a - x_p).invert().unwrap();
        let x_r = lambda1 * lambda1 - x_a - x_p;
        let lambda2 = C::Base::from_u64(2) * y_a * (x_a - x_r).invert().unwrap() - lambda1;
        region.assign_advice(|| "lambda1", config.lambda1, row, || Ok(lambda1))?;
        region.assign_advice(|| "lambda2", config.lambda2, row, || Ok(lambda2))?;

        // Compute and assign `x_a` for the next row
        let x_a_new = lambda2 * lambda2 - x_a - x_r;
        y_a = lambda2 * (x_a - x_a_new) - y_a;
        x_a = x_a_new;
        x_a_cell = region.assign_advice(|| "x_a", config.x_a, row + 1, || Ok(x_a))?;
    }

    for (row, k) in complete_range.zip(k_complete.iter()) {
        let y_a_cell = region.assign_advice(|| "y_a", config.y_a, row, || Ok(y_a))?;
        acc = EccPoint {
            x: CellValue::new(x_a_cell, Some(x_a)),
            y: CellValue::new(y_a_cell, Some(y_a)),
        };

        let x_p_val = base.x.value.unwrap();
        let x_p_cell = region.assign_advice(|| "x_p", config.x_p, row, || Ok(x_p_val))?;

        let mut y_p_val = base.y.value.unwrap();
        if k.value.unwrap() == C::Base::zero() {
            y_p_val = -y_p_val;
        }
        let y_p_cell = region.assign_advice(|| "y_p", config.y_p, row, || Ok(y_p_val))?;
        let p = EccPoint {
            x: CellValue::new(x_p_cell, Some(x_p_val)),
            y: CellValue::new(y_p_cell, Some(y_p_val)),
        };

        let tmp_acc = add_complete::assign_region(&p, &acc, region, config.clone()).unwrap();
        acc = add_complete::assign_region(&acc, &tmp_acc, region, config.clone()).unwrap();
    }

    let x_p_val = base.x.value.unwrap();
    let x_p_cell = region.assign_advice(|| "x_p", config.x_p, k_0_row, || Ok(x_p_val))?;

    let mut y_p_val = base.y.value.unwrap();
    if k_0.value.unwrap() == C::Base::zero() {
        y_p_val = -y_p_val;
    }

    let y_p_cell = region.assign_advice(|| "y_p", config.y_p, k_0_row, || Ok(y_p_val))?;
    let p = EccPoint {
        x: CellValue::new(x_p_cell, Some(x_p_val)),
        y: CellValue::new(y_p_cell, Some(y_p_val)),
    };

    acc = add_complete::assign_region(&acc, &p, region, config.clone()).unwrap();

    Ok(acc)
}
