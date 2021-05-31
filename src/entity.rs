/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::layer::Layer;
use crate::component::Component;
use crate::scene::Scene;
use std::sync::{RwLock, Arc};
use crate::engine::Engine;
use crate::engine_object::EngineObject;

pub struct Entity {
    pub name: String,
    /** The layer the entity belongs to. */
    pub layer: Layer,

    /** @internal */
    pub _is_active_in_hierarchy: bool,
    /** @internal */
    pub _components: Vec<Arc<RwLock<dyn Component>>>,
    /** @internal */
    pub _children: Vec<Arc<RwLock<Entity>>>,
    /** @internal */
    pub _scene: Option<Scene>,
    /** @internal */
    pub _is_root: bool,
    /** @internal */
    pub _is_active: bool,

    _parent: Option<Arc<RwLock<Entity>>>,
    _active_changed_components: Vec<Arc<RwLock<dyn Component>>>,

    _engine: Arc<RwLock<Engine>>,
}

impl Entity {
    pub fn new(engine: Arc<RwLock<Engine>>, name: Option<String>) -> Entity {
        return Entity {
            name: name.unwrap_or("".to_string()),
            layer: Layer::Layer0,
            _is_active_in_hierarchy: false,
            _components: vec![],
            _children: vec![],
            _scene: None,
            _is_root: false,
            _is_active: false,
            _parent: None,
            _active_changed_components: vec![],
            _engine: engine,
        };
    }
}

impl Entity {

}

impl EngineObject for Entity {
    fn engine(&self) -> &Arc<RwLock<Engine>> {
        return &self._engine;
    }

    fn engine_mut(&mut self) -> &mut Arc<RwLock<Engine>> {
        return &mut self._engine;
    }
}