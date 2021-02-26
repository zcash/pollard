use std::{collections::HashMap, marker::PhantomData};

use super::{EccInstructions, FixedPoints};
use halo2::{
    arithmetic::{CurveAffine, FieldExt},
    circuit::{Cell, Chip, Layouter},
    plonk::{Advice, Column, ConstraintSystem, Error, Fixed, Permutation, Selector},
};

/// Configuration for the ECC chip
#[derive(Clone, Debug)]
pub struct EccConfig {
    num_windows: usize,
    window_width: usize,
    num_complete_bits: usize,
    k: Column<Advice>,
    x_a: Column<Advice>,
    y_a: Column<Advice>,
    x_p: Column<Advice>,
    y_p: Column<Advice>,
    u: Column<Advice>,
    lambda1: Column<Advice>,
    lambda2: Column<Advice>,
    add_complete_inv: [Column<Advice>; 4],
    add_complete_bool: [Column<Advice>; 4],
    lagrange_coeffs: [Column<Fixed>; 8],
    fixed_z: Column<Fixed>,
    q_add: Selector,
    q_add_complete: Selector,
    q_double: Selector,
    q_mul: Selector,
    q_mul_fixed: Selector,
    q_point: Selector,
    q_scalar_var: Selector,
    q_scalar_fixed: Selector,
    perm_scalar: Permutation,
    perm_sum: Permutation,
}

/// A chip implementing EccInstructions
#[derive(Debug)]
pub struct EccChip<C: CurveAffine> {
    _marker: PhantomData<C>,
}

impl EccConfig {
    fn configure<C: CurveAffine>(
        meta: &mut ConstraintSystem<C::Base>,
        num_windows: usize,
        window_width: usize,
        num_complete_bits: usize,
        k: Column<Advice>,
        x_a: Column<Advice>,
        y_a: Column<Advice>,
        x_p: Column<Advice>,
        y_p: Column<Advice>,
        u: Column<Advice>,
        lambda1: Column<Advice>,
        lambda2: Column<Advice>,
        add_complete_inv: [Column<Advice>; 4],
        add_complete_bool: [Column<Advice>; 4],
    ) -> EccConfig {
        let q_add = meta.selector();
        let q_add_complete = meta.selector();
        let q_double = meta.selector();
        let q_mul = meta.selector();
        let q_mul_fixed = meta.selector();
        let q_point = meta.selector();
        let q_scalar_var = meta.selector();
        let q_scalar_fixed = meta.selector();

        let lagrange_coeffs = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];
        let fixed_z = meta.fixed_column();

        // Copy decomposed scalar
        let perm_scalar = Permutation::new(meta, &[k.into()]);

        // Initialise some accumulator from a witnessed point
        let perm_sum = Permutation::new(meta, &[x_p.into(), y_p.into(), x_a.into(), y_a.into()]);

        EccConfig {
            num_windows,
            window_width,
            num_complete_bits,
            k,
            x_a,
            y_a,
            x_p,
            y_p,
            u,
            lambda1,
            lambda2,
            add_complete_inv,
            add_complete_bool,
            lagrange_coeffs,
            fixed_z,
            q_add,
            q_add_complete,
            q_double,
            q_mul,
            q_mul_fixed,
            q_point,
            q_scalar_var,
            q_scalar_fixed,
            perm_scalar,
            perm_sum,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EccLoaded<C: CurveAffine> {
    lagrange_coeffs: Option<HashMap<OrchardFixedPoints<C>, Vec<Vec<C::Base>>>>,
    z: Option<HashMap<OrchardFixedPoints<C>, Vec<C::Base>>>,
}

impl<C: CurveAffine> EccLoaded<C> {
    fn get_lagrange_coeffs(&self, point: OrchardFixedPoints<C>) -> Option<Vec<Vec<C::Base>>> {
        if let Some(lagrange_coeffs) = &self.lagrange_coeffs {
            lagrange_coeffs.get(&point).cloned()
        } else {
            None
        }
    }

    fn get_z(&self, point: OrchardFixedPoints<C>) -> Option<Vec<C::Base>> {
        if let Some(z) = &self.z {
            z.get(&point).cloned()
        } else {
            None
        }
    }
}

impl<C: CurveAffine> Chip for EccChip<C> {
    type Config = EccConfig;
    type Field = C::Base;
    type Loaded = EccLoaded<C>;

    fn load(layouter: &mut impl Layouter<Self>) -> Result<Self::Loaded, Error> {
        // Load fixed bases (interpolation polynomials)
        todo!()
    }
}

// enum containing each fixed point type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchardFixedPoints<C: CurveAffine> {
    NullifierBase(C::CurveExt),
    NoteCommitBase(C::CurveExt),
    ValueCommitV(C::CurveExt),
    ValueCommitR(C::CurveExt),
    CommitIvkBase(C::CurveExt),
}

impl<C: CurveAffine> OrchardFixedPoints<C> {
    fn value(&self) -> C::CurveExt {
        match *self {
            OrchardFixedPoints::NullifierBase(point) => point,
            OrchardFixedPoints::NoteCommitBase(point) => point,
            OrchardFixedPoints::ValueCommitV(point) => point,
            OrchardFixedPoints::ValueCommitR(point) => point,
            OrchardFixedPoints::CommitIvkBase(point) => point,
        }
    }
}

impl<C: CurveAffine> std::hash::Hash for OrchardFixedPoints<C> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match *self {
            OrchardFixedPoints::NullifierBase(_) => {
                state.write(&format!("{:?}", "NullifierBase").as_bytes())
            }
            OrchardFixedPoints::NoteCommitBase(_) => {
                state.write(&format!("{:?}", "NoteCommitBase").as_bytes())
            }
            OrchardFixedPoints::ValueCommitV(_) => {
                state.write(&format!("{:?}", "ValueCommitV").as_bytes())
            }
            OrchardFixedPoints::ValueCommitR(_) => {
                state.write(&format!("{:?}", "ValueCommitR").as_bytes())
            }
            OrchardFixedPoints::CommitIvkBase(_) => {
                state.write(&format!("{:?}", "CommitIvkBase").as_bytes())
            }
        }
    }
}

impl<C: CurveAffine> FixedPoints<C> for OrchardFixedPoints<C> {}

#[derive(Clone, Debug)]
pub(crate) struct CellValue<F: FieldExt> {
    cell: Cell,
    value: Option<F>,
}

impl<F: FieldExt> CellValue<F> {
    pub fn new(cell: Cell, value: Option<F>) -> Self {
        CellValue { cell, value }
    }
}

#[derive(Clone, Debug)]
pub struct EccScalarVar<C: CurveAffine> {
    value: Option<C::Scalar>,
    k_bits: Vec<CellValue<C::Base>>, // bitwise decomposition
}

#[derive(Clone, Debug)]
pub struct EccScalarFixed<C: CurveAffine> {
    value: Option<C::Scalar>,
    k_bits: Vec<CellValue<C::Base>>, // 3-bit decomposition
}

#[derive(Clone, Debug)]
pub struct EccPoint<F: FieldExt> {
    x: CellValue<F>,
    y: CellValue<F>,
}

#[derive(Clone, Debug)]
pub struct EccFixedPoint<C: CurveAffine> {
    fixed_point: OrchardFixedPoints<C>,
    lagrange_coeffs: Option<Vec<Vec<C::Base>>>,
    z: Option<Vec<C::Base>>,
}

impl<C: CurveAffine> EccInstructions<C> for EccChip<C> {
    type ScalarVar = EccScalarVar<C>;
    type ScalarFixed = EccScalarFixed<C>;
    type Point = EccPoint<C::Base>;
    type FixedPoint = EccFixedPoint<C>;
    type FixedPoints = OrchardFixedPoints<C>;

    /// Witnesses the given scalar as a private input to the circuit for variable-based scalar mul.
    fn witness_scalar_var(
        layouter: &mut impl Layouter<Self>,
        value: Option<C::Scalar>,
    ) -> Result<Self::ScalarVar, Error> {
        todo!()
    }

    /// Witnesses the given scalar as a private input to the circuit for fixed-based scalar mul.
    fn witness_scalar_fixed(
        layouter: &mut impl Layouter<Self>,
        value: Option<C::Scalar>,
    ) -> Result<Self::ScalarFixed, Error> {
        todo!()
    }

    fn witness_point(
        layouter: &mut impl Layouter<Self>,
        value: Option<C::CurveExt>,
    ) -> Result<Self::Point, Error> {
        todo!()
    }

    fn get_fixed(
        layouter: &mut impl Layouter<Self>,
        fixed_point: Self::FixedPoints,
    ) -> Result<Self::FixedPoint, Error> {
        let loaded = layouter.loaded();

        let lagrange_coeffs = loaded.get_lagrange_coeffs(fixed_point.clone());
        let z = loaded.get_z(fixed_point.clone());
        Ok(EccFixedPoint {
            fixed_point,
            lagrange_coeffs,
            z,
        })
    }

    fn add(
        layouter: &mut impl Layouter<Self>,
        a: &Self::Point,
        b: &Self::Point,
    ) -> Result<Self::Point, Error> {
        todo!()
    }

    fn add_complete(
        layouter: &mut impl Layouter<Self>,
        a: &Self::Point,
        b: &Self::Point,
    ) -> Result<Self::Point, Error> {
        todo!()
    }

    fn double(layouter: &mut impl Layouter<Self>, a: &Self::Point) -> Result<Self::Point, Error> {
        todo!()
    }

    fn mul(
        layouter: &mut impl Layouter<Self>,
        scalar: &Self::ScalarVar,
        base: &Self::Point,
    ) -> Result<Self::Point, Error> {
        todo!()
    }

    fn mul_fixed(
        layouter: &mut impl Layouter<Self>,
        scalar: &Self::ScalarFixed,
        base: &Self::FixedPoint,
    ) -> Result<Self::Point, Error> {
        todo!()
    }
}
