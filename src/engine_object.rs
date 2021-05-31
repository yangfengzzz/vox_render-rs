/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::engine::Engine;
use std::sync::{RwLock, Arc};

pub trait EngineObject {
    fn engine(&self) -> &Arc<RwLock<Engine>>;

    fn engine_mut(&mut self) -> &mut Arc<RwLock<Engine>>;
}