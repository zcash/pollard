use super::super::EccInstructions;
use super::{CellValue, EccChip, EccPoint};

use halo2::{
    arithmetic::{CurveAffine, Field, FieldExt},
    circuit::{Chip, Region},
    plonk::{ConstraintSystem, Error, Expression},
};

pub(super) fn create_gate<C: CurveAffine>(
    meta: &mut ConstraintSystem<C::Base>,
    q_double: Expression<C::Base>,
    x_a: Expression<C::Base>,
    y_a: Expression<C::Base>,
    x_p: Expression<C::Base>,
    y_p: Expression<C::Base>,
) {
    // first expression
    meta.create_gate("point doubling expr1", |_| {
        let x_p_4 = x_p.clone() * x_p.clone() * x_p.clone() * x_p.clone();
        let expr1 = y_p.clone()
            * y_p.clone()
            * (x_a.clone() + x_p.clone() * C::Base::from_u64(2))
            * C::Base::from_u64(2)
            - x_p_4 * C::Base::from_u64(9);
        q_double.clone() * expr1
    });

    // second expression
    meta.create_gate("point doubling expr2", |_| {
        let expr2 = y_p.clone() * (y_a + y_p) * C::Base::from_u64(2)
            - x_p.clone() * x_p.clone() * (x_p - x_a) * C::Base::from_u64(3);

        q_double * expr2
    });
}

pub(super) fn assign_region<C: CurveAffine>(
    a: &<EccChip<C> as EccInstructions<C>>::Point,
    region: &mut Region<'_, EccChip<C>>,
    config: <EccChip<C> as Chip>::Config,
) -> Result<EccPoint<C::Base>, Error> {
    let x_p_val = a.x.value;
    let y_p_val = a.y.value;

    config.q_double.enable(region, 0)?;
    region.assign_advice(
        || "x_p",
        config.x_p,
        0,
        || x_p_val.ok_or(Error::SynthesisError),
    )?;
    region.assign_advice(
        || "y_p",
        config.y_p,
        0,
        || y_p_val.ok_or(Error::SynthesisError),
    )?;

    let lambda1 = x_p_val.zip(y_p_val).map(|(x_p, y_p)| {
        C::Base::from_u64(3) * x_p * x_p * (C::Base::from_u64(2) * y_p).invert().unwrap()
    });

    let x_a_val = lambda1
        .zip(x_p_val)
        .map(|(lambda1, x_p)| lambda1 * lambda1 * C::Base::from_u64(2) * x_p);
    let x_a_var = region.assign_advice(
        || "x_a_val",
        config.x_a,
        0,
        || x_a_val.ok_or(Error::SynthesisError),
    )?;

    let y_a_val = lambda1
        .zip(x_p_val)
        .zip(x_a_val)
        .zip(y_p_val)
        .map(|(((lambda1, x_p), x_a), y_p)| lambda1 * (x_p - x_a) - y_p);
    let y_a_var = region.assign_advice(
        || "y_a_val",
        config.y_a,
        0,
        || y_a_val.ok_or(Error::SynthesisError),
    )?;

    Ok(EccPoint {
        x: CellValue::new(x_a_var, x_a_val),
        y: CellValue::new(y_a_var, y_a_val),
    })
}
