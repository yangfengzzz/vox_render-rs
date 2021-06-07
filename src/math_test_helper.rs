/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

pub const K_FLOAT_NEAR_TOLERANCE: f32 = 1.0e-5;
pub const K_FLOAT_NEAR_EST_TOLERANCE: f32 = 1.0e-3;

#[macro_export]
macro_rules! expect_near {
    ($x:expr, $y:expr, $d:expr) => {
        if ($x - $y).abs() > $d { panic!(); }
    }
}

// Implements "float near" test as a function. Avoids overloading compiler
// optimizer when too much expect_near are used in a single compilation unit.
pub fn expect_float_near(_a: f32, _b: f32, _tol: Option<f32>) {
    expect_near!(_a, _b, _tol.unwrap_or(K_FLOAT_NEAR_TOLERANCE));
}

// Implements "int equality" test as a function. Avoids overloading compiler
// optimizer when too much EXPECT_TRUE are used in a single compilation unit.
pub fn expect_int_eq(_a: i32, _b: i32) {
    assert_eq!(_a, _b);
}

// Implements "bool equality" test as a function. Avoids overloading compiler
// optimizer when too much EXPECT_TRUE are used in a single compilation unit.
pub fn expect_true(_b: bool) {
    assert_eq!(_b, true);
}

// Macro for testing floats, dedicated to estimated functions with a lower
// precision.
#[macro_export]
macro_rules! expect_float_eq_est {
    ($_expected:expr, $_x:expr) => {
        expect_near!($_expected, $_x, K_FLOAT_NEAR_EST_TOLERANCE);
    };
}

// Macro for testing ozz::math::Float4 members with x, y, z, w float values,
// using EXPECT_FLOAT_EQ internally.
#[macro_export]
macro_rules! expect_float4_eq {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_w:expr) => {
        {
            expect_float_near($expected.x, $_x, None);
            expect_float_near($expected.y, $_y, None);
            expect_float_near($expected.z, $_z, None);
            expect_float_near($expected.w, $_w, None);
        }
    };
}

// Macro for testing ozz::math::Float3 members with x, y, z float values,
// using EXPECT_FLOAT_EQ internally.
#[macro_export]
macro_rules! expect_float3_eq {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr) => {
        {
            expect_float_near($expected.x, $_x, None);
            expect_float_near($expected.y, $_y, None);
            expect_float_near($expected.z, $_z, None);
        }
    };
}

// Macro for testing ozz::math::Float2 members with x, y float values,
// using EXPECT_NEAR internally.
#[macro_export]
macro_rules! expect_float2_eq {
    ($expected:expr, $_x:expr, $_y:expr) => {
        {
            expect_float_near($expected.x, $_x, None);
            expect_float_near($expected.y, $_y, None);
        }
    };
}

// Macro for testing ozz::math::Quaternion members with x, y, z, w float value.
#[macro_export]
macro_rules! expect_quaternion_eq {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_w:expr) => {
        {
            expect_float_near($expected.x, $_x, None);
            expect_float_near($expected.y, $_y, None);
            expect_float_near($expected.z, $_z, None);
            expect_float_near($expected.w, $_w, None);
        }
    };
}

//--------------------------------------------------------------------------------------------------
#[macro_export]
macro_rules! _impl_expect_simd_float_eq_tol {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_w:expr, $_tol:expr) => {
        unsafe {
            let u = SimdFloat4Union {
                ret: $expected.data,
            };

            expect_float_near(u.af[0], $_x, Some($_tol));
            expect_float_near(u.af[1], $_y, Some($_tol));
            expect_float_near(u.af[2], $_z, Some($_tol));
            expect_float_near(u.af[3], $_w, Some($_tol));
        }
    };
}

#[macro_export]
macro_rules! _impl_expect_simd_float_eq {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_w:expr) => {
        _impl_expect_simd_float_eq_tol!($expected, $_x, $_y, $_z, $_w, K_FLOAT_NEAR_TOLERANCE);
    };
}

#[macro_export]
macro_rules! _impl_expect_simd_float_eq_est {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_w:expr) => {
        _impl_expect_simd_float_eq_tol!($expected, $_x, $_y, $_z, $_w, K_FLOAT_NEAR_EST_TOLERANCE);
    };
}

// Macro for testing ozz::math::simd::SimdFloat members with x, y, z, w values.
#[macro_export]
macro_rules! expect_simd_float_eq {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_w:expr) => {
        _impl_expect_simd_float_eq!($expected, $_x, $_y, $_z, $_w);
    };
}

// Macro for testing ozz::math::simd::SimdFloat members with x, y, z, w values.
// Dedicated to estimated functions with a lower precision.
#[macro_export]
macro_rules! expect_simd_float_eq_est {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_w:expr) => {
        _impl_expect_simd_float_eq_est!($expected, $_x, $_y, $_z, $_w);
    };
}

//--------------------------------------------------------------------------------------------------
// Macro for testing ozz::math::simd::SimdFloat members with x, y values with
// a user defined precision.
#[macro_export]
macro_rules! _impl_expect_simd_float2_eq_tol {
    ($expected:expr, $_x:expr, $_y:expr, $_tol:expr) => {
        unsafe {
            let u = SimdFloat4Union {
                ret: $expected.data,
            };

            expect_float_near(u.af[0], $_x, Some($_tol));
            expect_float_near(u.af[1], $_y, Some($_tol));
        }
    };
}

// Macro for testing ozz::math::simd::SimdFloat members with x, y values.
#[macro_export]
macro_rules! expect_simd_float2_eq {
    ($expected:expr, $_x:expr, $_y:expr) => {
        _impl_expect_simd_float2_eq_tol!($expected, $_x, $_y, K_FLOAT_NEAR_TOLERANCE);
    };
}

// Macro for testing ozz::math::simd::SimdFloat members with x, y values.
// Dedicated to estimated functions with a lower precision.
#[macro_export]
macro_rules! expect_simd_float2_eq_est {
    ($expected:expr, $_x:expr, $_y:expr) => {
        _impl_expect_simd_float2_eq_tol!($expected, $_x, $_y, K_FLOAT_NEAR_EST_TOLERANCE);
    };
}

//--------------------------------------------------------------------------------------------------
// Macro for testing ozz::math::simd::SimdFloat members with x, y, z values with
// a user defined precision.
#[macro_export]
macro_rules! _impl_expect_simd_float3_eq_tol {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_tol:expr) => {
        unsafe {
            let u = SimdFloat4Union {
                ret: $expected.data,
            };

            expect_float_near(u.af[0], $_x, Some($_tol));
            expect_float_near(u.af[1], $_y, Some($_tol));
            expect_float_near(u.af[2], $_z, Some($_tol));
        }
    };
}

// Macro for testing ozz::math::simd::SimdFloat members with x, y, z values.
#[macro_export]
macro_rules! expect_simd_float3_eq {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr) => {
        _impl_expect_simd_float3_eq_tol!($expected, $_x, $_y, $_z, K_FLOAT_NEAR_TOLERANCE);
    };
}

// Macro for testing ozz::math::simd::SimdFloat members with x, y, z values.
// Dedicated to estimated functions with a lower precision.
#[macro_export]
macro_rules! expect_simd_float3_eq_est {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr) => {
        _impl_expect_simd_float3_eq_tol!($expected, $_x, $_y, $_z, K_FLOAT_NEAR_EST_TOLERANCE);
    };
}

//--------------------------------------------------------------------------------------------------
// Macro for testing ozz::math::simd::SimdInt members with x, y, z, w values.
#[macro_export]
macro_rules! expect_simd_int_eq {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_w:expr) => {
        unsafe {
            let u = SimdInt4Union {
                ret: $expected.data,
            };

            expect_int_eq(u.af[0], $_x);
            expect_int_eq(u.af[1], $_y);
            expect_int_eq(u.af[2], $_z);
            expect_int_eq(u.af[3], $_w);
        }
    };
}

#[macro_export]
macro_rules! expect_float4x4_eq {
    ($expected:expr, $_x0:expr, $_x1:expr, $_x2:expr, $_x3:expr, $_y0:expr, $_y1:expr, $_y2:expr, $_y3:expr,
    $_z0:expr, $_z1:expr, $_z2:expr, $_z3:expr, $_w0:expr, $_w1:expr, $_w2:expr, $_w3:expr) => {
        _impl_expect_simd_float_eq!($expected.cols[0], $_x0, $_x1, $_x2, $_x3);
        _impl_expect_simd_float_eq!($expected.cols[1], $_y0, $_y1, $_y2, $_y3);
        _impl_expect_simd_float_eq!($expected.cols[2], $_z0, $_z1, $_z2, $_z3);
        _impl_expect_simd_float_eq!($expected.cols[3], $_w0, $_w1, $_w2, $_w3);
    };
}

// Macro for testing ozz::math::simd::SimdQuaternion members with x, y, z, w
// values.
#[macro_export]
macro_rules! expect_simd_quaternion_eq {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_w:expr) => {
        _impl_expect_simd_float_eq!($expected.xyzw, $_x, $_y, $_z, $_w);
    };
}

// Macro for testing ozz::math::simd::SimdQuaternion members with x, y, z, w
// values.
#[macro_export]
macro_rules! expect_simd_quaternion_eq_est {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_w:expr) => {
        _impl_expect_simd_float_eq_est!($expected.xyzw, $_x, $_y, $_z, $_w);
    };
}

// Macro for testing ozz::math::simd::SimdQuaternion members with x, y, z, w
// values.
#[macro_export]
macro_rules! expect_simd_quaternion_eq_tol {
    ($expected:expr, $_x:expr, $_y:expr, $_z:expr, $_w:expr, $_tol:expr) => {
        _impl_expect_simd_float_eq_tol!($expected.xyzw, $_x, $_y, $_z, $_w, $_tol);
    };
}
//--------------------------------------------------------------------------------------------------
// Macro for testing ozz::math::SoaFloat4 members with x, y, z, w float values.
#[macro_export]
macro_rules! expect_soa_float4_eq {
    ($expected:expr, $_x0:expr, $_x1:expr, $_x2:expr, $_x3:expr, $_y0:expr, $_y1:expr, $_y2:expr, $_y3:expr,
    $_z0:expr, $_z1:expr, $_z2:expr, $_z3:expr, $_w0:expr, $_w1:expr, $_w2:expr, $_w3:expr) => {
        _impl_expect_simd_float_eq!($expected.x, $_x0, $_x1, $_x2, $_x3);
        _impl_expect_simd_float_eq!($expected.y, $_y0, $_y1, $_y2, $_y3);
        _impl_expect_simd_float_eq!($expected.z, $_z0, $_z1, $_z2, $_z3);
        _impl_expect_simd_float_eq!($expected.w, $_w0, $_w1, $_w2, $_w3);
    };
}

// Macro for testing ozz::math::SoaFloat4 members with x, y, z, w float values.
// Dedicated to estimated functions with a lower precision.
#[macro_export]
macro_rules! expect_soa_float4_eq_est {
    ($expected:expr, $_x0:expr, $_x1:expr, $_x2:expr, $_x3:expr, $_y0:expr, $_y1:expr, $_y2:expr, $_y3:expr,
    $_z0:expr, $_z1:expr, $_z2:expr, $_z3:expr, $_w0:expr, $_w1:expr, $_w2:expr, $_w3:expr) => {
        _impl_expect_simd_float_eq_est!($expected.x, $_x0, $_x1, $_x2, $_x3);
        _impl_expect_simd_float_eq_est!($expected.y, $_y0, $_y1, $_y2, $_y3);
        _impl_expect_simd_float_eq_est!($expected.z, $_z0, $_z1, $_z2, $_z3);
        _impl_expect_simd_float_eq_est!($expected.w, $_w0, $_w1, $_w2, $_w3);
    };
}

//--------------------------------------------------------------------------------------------------
// Macro for testing ozz::math::SoaFloat3 members with x, y, z float values.
#[macro_export]
macro_rules! expect_soa_float3_eq {
    ($expected:expr, $_x0:expr, $_x1:expr, $_x2:expr, $_x3:expr, $_y0:expr, $_y1:expr, $_y2:expr, $_y3:expr,
    $_z0:expr, $_z1:expr, $_z2:expr, $_z3:expr) => {
        _impl_expect_simd_float_eq!($expected.x, $_x0, $_x1, $_x2, $_x3);
        _impl_expect_simd_float_eq!($expected.y, $_y0, $_y1, $_y2, $_y3);
        _impl_expect_simd_float_eq!($expected.z, $_z0, $_z1, $_z2, $_z3);
    };
}

// Macro for testing ozz::math::SoaFloat3 members with x, y, z float values.
// Dedicated to estimated functions with a lower precision.
#[macro_export]
macro_rules! expect_soa_float3_eq_est {
    ($expected:expr, $_x0:expr, $_x1:expr, $_x2:expr, $_x3:expr, $_y0:expr, $_y1:expr, $_y2:expr, $_y3:expr,
    $_z0:expr, $_z1:expr, $_z2:expr, $_z3:expr) => {
        _impl_expect_simd_float_eq_est!($expected.x, $_x0, $_x1, $_x2, $_x3);
        _impl_expect_simd_float_eq_est!($expected.y, $_y0, $_y1, $_y2, $_y3);
        _impl_expect_simd_float_eq_est!($expected.z, $_z0, $_z1, $_z2, $_z3);
    };
}

//--------------------------------------------------------------------------------------------------
// Macro for testing ozz::math::SoaFloat2 members with x, y float values.
#[macro_export]
macro_rules! expect_soa_float2_eq {
    ($expected:expr, $_x0:expr, $_x1:expr, $_x2:expr, $_x3:expr, $_y0:expr, $_y1:expr, $_y2:expr, $_y3:expr) => {
        _impl_expect_simd_float_eq!($expected.x, $_x0, $_x1, $_x2, $_x3);
        _impl_expect_simd_float_eq!($expected.y, $_y0, $_y1, $_y2, $_y3);
    };
}

// Macro for testing ozz::math::SoaFloat2 members with x, y float values.
// Dedicated to estimated functions with a lower precision.
#[macro_export]
macro_rules! expect_soa_float2_eq_est {
    ($expected:expr, $_x0:expr, $_x1:expr, $_x2:expr, $_x3:expr, $_y0:expr, $_y1:expr, $_y2:expr, $_y3:expr) => {
        _impl_expect_simd_float_eq_est!($expected.x, $_x0, $_x1, $_x2, $_x3);
        _impl_expect_simd_float_eq_est!($expected.y, $_y0, $_y1, $_y2, $_y3);
    };
}

//--------------------------------------------------------------------------------------------------
// Macro for testing ozz::math::SoaFloat1 members with x float values.
#[macro_export]
macro_rules! expect_soa_float1_eq {
    ($expected:expr, $_x0:expr, $_x1:expr, $_x2:expr, $_x3:expr) => {
        _impl_expect_simd_float_eq!($expected.x, $_x0, $_x1, $_x2, $_x3);
    };
}

// Macro for testing ozz::math::SoaFloat1 members with x float values.
// Dedicated to estimated functions with a lower precision.
#[macro_export]
macro_rules! expect_soa_float1_eq_est {
    ($expected:expr, $_x0:expr, $_x1:expr, $_x2:expr, $_x3:expr) => {
        _impl_expect_simd_float_eq_est!($expected.x, $_x0, $_x1, $_x2, $_x3);
    };
}

//--------------------------------------------------------------------------------------------------
// Macro for testing ozz::math::SoaQuaternion members with x, y, z, w float values.
#[macro_export]
macro_rules! expect_soa_quaternion_eq {
    ($expected:expr, $_x0:expr, $_x1:expr, $_x2:expr, $_x3:expr, $_y0:expr, $_y1:expr, $_y2:expr, $_y3:expr,
    $_z0:expr, $_z1:expr, $_z2:expr, $_z3:expr, $_w0:expr, $_w1:expr, $_w2:expr, $_w3:expr) => {
        _impl_expect_simd_float_eq!($expected.x, $_x0, $_x1, $_x2, $_x3);
        _impl_expect_simd_float_eq!($expected.y, $_y0, $_y1, $_y2, $_y3);
        _impl_expect_simd_float_eq!($expected.z, $_z0, $_z1, $_z2, $_z3);
        _impl_expect_simd_float_eq!($expected.w, $_w0, $_w1, $_w2, $_w3);
    };
}

// Macro for testing ozz::math::SoaQuaternion members with x, y, z, w float values.
// Dedicated to estimated functions with a lower precision.
#[macro_export]
macro_rules! expect_soa_quaternion_eq_est {
    ($expected:expr, $_x0:expr, $_x1:expr, $_x2:expr, $_x3:expr, $_y0:expr, $_y1:expr, $_y2:expr, $_y3:expr,
    $_z0:expr, $_z1:expr, $_z2:expr, $_z3:expr, $_w0:expr, $_w1:expr, $_w2:expr, $_w3:expr) => {
        _impl_expect_simd_float_eq_est!($expected.x, $_x0, $_x1, $_x2, $_x3);
        _impl_expect_simd_float_eq_est!($expected.y, $_y0, $_y1, $_y2, $_y3);
        _impl_expect_simd_float_eq_est!($expected.z, $_z0, $_z1, $_z2, $_z3);
        _impl_expect_simd_float_eq_est!($expected.w, $_w0, $_w1, $_w2, $_w3);
    };
}

//--------------------------------------------------------------------------------------------------
#[macro_export]
macro_rules! expect_soa_float4x4_eq {
    ($expected:expr, $col0xx:expr, $col0xy:expr, $col0xz:expr, $col0xw:expr, $col0yx:expr, $col0yy:expr, $col0yz:expr, $col0yw:expr,
    $col0zx:expr, $col0zy:expr, $col0zz:expr, $col0zw:expr, $col0wx:expr, $col0wy:expr, $col0wz:expr, $col0ww:expr, $col1xx:expr,
    $col1xy:expr, $col1xz:expr, $col1xw:expr, $col1yx:expr, $col1yy:expr, $col1yz:expr, $col1yw:expr, $col1zx:expr, $col1zy:expr,
    $col1zz:expr, $col1zw:expr, $col1wx:expr, $col1wy:expr, $col1wz:expr, $col1ww:expr, $col2xx:expr, $col2xy:expr, $col2xz:expr,
    $col2xw:expr, $col2yx:expr, $col2yy:expr, $col2yz:expr, $col2yw:expr, $col2zx:expr, $col2zy:expr, $col2zz:expr, $col2zw:expr,
    $col2wx:expr, $col2wy:expr, $col2wz:expr, $col2ww:expr, $col3xx:expr, $col3xy:expr, $col3xz:expr, $col3xw:expr, $col3yx:expr,
    $col3yy:expr, $col3yz:expr, $col3yw:expr, $col3zx:expr, $col3zy:expr, $col3zz:expr, $col3zw:expr, $col3wx:expr, $col3wy:expr,
    $col3wz:expr, $col3ww:expr) => {
        _impl_expect_simd_float_eq!($expected.cols[0].x, $col0xx, $col0xy, $col0xz, $col0xw);
        _impl_expect_simd_float_eq!($expected.cols[0].y, $col0yx, $col0yy, $col0yz, $col0yw);
        _impl_expect_simd_float_eq!($expected.cols[0].z, $col0zx, $col0zy, $col0zz, $col0zw);
        _impl_expect_simd_float_eq!($expected.cols[0].w, $col0wx, $col0wy, $col0wz, $col0ww);
        _impl_expect_simd_float_eq!($expected.cols[1].x, $col1xx, $col1xy, $col1xz, $col1xw);
        _impl_expect_simd_float_eq!($expected.cols[1].y, $col1yx, $col1yy, $col1yz, $col1yw);
        _impl_expect_simd_float_eq!($expected.cols[1].z, $col1zx, $col1zy, $col1zz, $col1zw);
        _impl_expect_simd_float_eq!($expected.cols[1].w, $col1wx, $col1wy, $col1wz, $col1ww);
        _impl_expect_simd_float_eq!($expected.cols[2].x, $col2xx, $col2xy, $col2xz, $col2xw);
        _impl_expect_simd_float_eq!($expected.cols[2].y, $col2yx, $col2yy, $col2yz, $col2yw);
        _impl_expect_simd_float_eq!($expected.cols[2].z, $col2zx, $col2zy, $col2zz, $col2zw);
        _impl_expect_simd_float_eq!($expected.cols[2].w, $col2wx, $col2wy, $col2wz, $col2ww);
        _impl_expect_simd_float_eq!($expected.cols[3].x, $col3xx, $col3xy, $col3xz, $col3xw);
        _impl_expect_simd_float_eq!($expected.cols[3].y, $col3yx, $col3yy, $col3yz, $col3yw);
        _impl_expect_simd_float_eq!($expected.cols[3].z, $col3zx, $col3zy, $col3zz, $col3zw);
        _impl_expect_simd_float_eq!($expected.cols[3].w, $col3wx, $col3wy, $col3wz, $col3ww);
    };
}