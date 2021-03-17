//! Constants used in the Orchard protocol.

/// $\ell^\mathsf{Orchard}_\mathsf{base}$
pub(crate) const L_ORCHARD_BASE: usize = 255;

// SWU hash-to-curve personalizations

/// SWU hash-to-curve personalization for the spending key base point
pub const SPENDING_KEY_GENERATOR_PERSONALIZATION: &str = "z.cash:Orchard";

/// SWU hash-to-curve personalization for the group hash for key diversification
pub const KEY_DIVERSIFICATION_PERSONALIZATION: &str = "z.cash:Orchard-gd";

/// SWU hash-to-curve personalization for the value commitment generator
pub const VALUE_COMMITMENT_PERSONALIZATION: &str = "z.cash:Orchard-cv";

/// SWU hash-to-curve personalization for the note commitment generator
pub const NOTE_COMMITMENT_PERSONALIZATION: &str = "z.cash:Orchard-NoteCommit";

/// SWU hash-to-curve personalization for the IVK commitment generator
pub const COMMIT_IVK_PERSONALIZATION: &str = "z.cash:Orchard-CommitIvk";

/// SWU hash-to-curve personalization for the nullifier generator
pub const NULLIFIER_K_PERSONALIZATION: &str = "z.cash:Orchard-Nullifier-K";
