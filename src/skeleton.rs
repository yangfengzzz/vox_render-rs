/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::skeleton::Constants::KMaxJoints;

// Defines Skeleton constant values.
pub enum Constants {
    // Defines the maximum number of joints.
    // This is limited in order to control the number of bits required to store
    // a joint index. Limiting the number of joints also helps handling worst
    // size cases, like when it is required to allocate an array of joints on
    // the stack.
    KMaxJoints = 1024,

    // Defines the maximum number of SoA elements required to store the maximum
    // number of joints.
    KMaxSoAJoints = (KMaxJoints as isize + 3) / 4,

    // Defines the index of the parent of the root joint (which has no parent in
    // fact).
    KNoParent = -1,
}