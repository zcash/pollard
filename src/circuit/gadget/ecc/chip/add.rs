use super::super::EccInstructions;
use super::{CellValue, EccChip, EccPoint};

use halo2::{
    arithmetic::{CurveAffine, Field},
    circuit::{Chip, Region},
    plonk::{ConstraintSystem, Error, Expression},
};

pub(super) fn create_gate<C: CurveAffine>(
    meta: &mut ConstraintSystem<C::Base>,
    q_add: Expression<C::Base>,
    x_p: Expression<C::Base>,
    y_p: Expression<C::Base>,
    x_q: Expression<C::Base>,
    y_q: Expression<C::Base>,
    x_a: Expression<C::Base>,
    y_a: Expression<C::Base>,
) {
    // First expression
    meta.create_gate("point addition expr1", |_| {
        let expr1 = x_a.clone() + x_q.clone() + x_p.clone()
            - (y_p.clone() - y_q.clone()) * (y_p.clone() - y_q.clone());

        q_add.clone() * expr1
    });

    // Second expression
    meta.create_gate("point addition expr2", |_| {
        let expr2 = (y_a + y_q.clone()) * (x_p - x_q.clone()) - (y_p - y_q) * (x_q - x_a);

        q_add * expr2
    });
}

pub(super) fn assign_region<C: CurveAffine>(
    a: &<EccChip<C> as EccInstructions<C>>::Point,
    b: &<EccChip<C> as EccInstructions<C>>::Point,
    region: &mut Region<'_, EccChip<C>>,
    config: <EccChip<C> as Chip>::Config,
) -> Result<EccPoint<C::Base>, Error> {
    let (x_p, y_p) = (a.x.value, a.y.value);
    let (x_q, y_q) = (b.x.value, b.y.value);

    config.q_add.enable(region, 0)?;

    region.assign_advice(|| "x_p", config.x_p, 0, || x_p.ok_or(Error::SynthesisError))?;
    region.assign_advice(|| "y_p", config.y_p, 0, || y_p.ok_or(Error::SynthesisError))?;
    region.assign_advice(|| "x_q", config.x_a, 0, || x_q.ok_or(Error::SynthesisError))?;
    region.assign_advice(|| "y_q", config.y_a, 0, || y_q.ok_or(Error::SynthesisError))?;

    let lambda1 = y_p
        .zip(y_q)
        .zip(x_p)
        .zip(x_q)
        .map(|(((y_p, y_q), x_p), x_q)| (y_p - y_q) * (x_p - x_q).invert().unwrap());

    let x_a_val = lambda1
        .zip(x_q)
        .zip(x_p)
        .map(|((lambda1, x_q), x_p)| lambda1 * lambda1 - x_q - x_p);
    let x_a_var = region.assign_advice(
        || "x_a",
        config.x_a,
        1,
        || x_a_val.ok_or(Error::SynthesisError),
    )?;

    let y_a_val = lambda1
        .zip(x_q)
        .zip(x_a_val)
        .zip(y_q)
        .map(|(((lambda1, x_q), x_a), y_q)| lambda1 * (x_q - x_a) - y_q);
    let y_a_var = region.assign_advice(
        || "y_a",
        config.y_a,
        1,
        || y_a_val.ok_or(Error::SynthesisError),
    )?;

    Ok(EccPoint {
        x: CellValue::new(x_a_var, x_a_val),
        y: CellValue::new(y_a_var, y_a_val),
    })
}
