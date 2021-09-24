/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::cmp::Ordering;

#[derive(Copy, Clone, Default)]
pub struct Entity {
    m_identity: u32,
}

impl Entity {
    pub(crate) fn new(identity: u32) -> Entity {
        return Entity {
            m_identity: identity
        };
    }
}

impl Entity {
    pub fn is_null(&self) -> bool {
        return self.m_identity == 0;
    }

    // an id that can be used for debugging/printing
    pub fn get_id(&self) -> u32 {
        return self.m_identity;
    }

    pub fn clear(&mut self) { self.m_identity = 0; }
}

impl PartialOrd for Entity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        return if self.m_identity < other.m_identity {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        };
    }
}

impl PartialEq for Entity {
    fn eq(&self, other: &Self) -> bool {
        return self.m_identity == other.m_identity;
    }
}