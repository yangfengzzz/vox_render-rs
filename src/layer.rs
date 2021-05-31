/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

/**
 * Layer, used for bit operations.
 */
pub enum Layer {
    Layer0 = 0x1,
    Layer1 = 0x2,
    Layer2 = 0x4,
    Layer3 = 0x8,
    Layer4 = 0x10,
    Layer5 = 0x20,
    Layer6 = 0x40,
    Layer7 = 0x80,
    Layer8 = 0x100,
    Layer9 = 0x200,
    Layer10 = 0x400,
    Layer11 = 0x800,
    Layer12 = 0x1000,
    Layer13 = 0x2000,
    Layer14 = 0x4000,
    Layer15 = 0x8000,
    Layer16 = 0x10000,
    Layer17 = 0x20000,
    Layer18 = 0x40000,
    Layer19 = 0x80000,
    Layer20 = 0x100000,
    Layer21 = 0x200000,
    Layer22 = 0x400000,
    Layer23 = 0x800000,
    Layer24 = 0x1000000,
    Layer25 = 0x2000000,
    Layer26 = 0x4000000,
    Layer27 = 0x8000000,
    Layer28 = 0x10000000,
    Layer29 = 0x20000000,
    Layer30 = 0x40000000,
    Layer31 = 0x80000000,
    Everything = 0xffffffff,
    Nothing = 0x0,
}
