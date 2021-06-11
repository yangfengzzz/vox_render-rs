/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

// Defines math trigonometric constants.
pub const K2PI: f32 = 6.283185307179586476925286766559;
pub const K_PI: f32 = std::f32::consts::PI;
pub const K_PI_2: f32 = std::f32::consts::FRAC_PI_2;
pub const K_PI_4: f32 = std::f32::consts::FRAC_PI_4;
pub const K_SQRT3: f32 = 1.7320508075688772935274463415059;
pub const K_SQRT3_2: f32 = 0.86602540378443864676372317075294;
pub const K_SQRT2: f32 = std::f32::consts::SQRT_2;
pub const K_SQRT2_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

// Angle unit conversion constants.
pub const K_DEGREE_TO_RADIAN: f32 = K_PI / 180.0;
pub const K_RADIAN_TO_DEGREE: f32 = 180.0 / K_PI;

// Defines the square normalization tolerance value.
pub const K_NORMALIZATION_TOLERANCE_SQ: f32 = 1e-6;
pub const K_NORMALIZATION_TOLERANCE_EST_SQ: f32 = 2e-3;

// Defines the square orthogonalisation tolerance value.
pub const K_ORTHOGONALISATION_TOLERANCE_SQ: f32 = 1e-16;