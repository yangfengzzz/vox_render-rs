/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::entity::Entity;

pub struct ComponentData {
    /** @internal */
    _entity: Entity,
    /** @internal */
    _destroyed: bool,

    _enabled: bool,
    _awake: bool,
}

pub trait Component {
    fn view(&self) -> &ComponentData;
    fn view_mut(&mut self) -> &mut ComponentData;

    /**
     * Indicates whether the component is enabled.
     */
    fn enabled(&self) -> bool {
        return self.view()._enabled;
    }

    fn set_enabled(&mut self, value: bool) {
        if value == self.view()._enabled {
            return;
        }
        self.view_mut()._enabled = value;
        if value {
            if self.view()._entity._is_active_in_hierarchy {
                self._on_enable();
            }
        } else {
            if self.view()._entity._is_active_in_hierarchy {
                self._on_disable();
            }
        }
    }

    /**
     * Indicates whether the component is destroyed.
     */
    fn destroyed(&self) -> bool {
        return self.view()._destroyed;
    }

    /**
     * Destroy this instance.
     */
    fn destroy(&mut self) {
        if self.view()._destroyed {
            return;
        }
        self._entity._removeComponent(this);
        if self.view()._entity._is_active_in_hierarchy {
            if self.view()._enabled {
                self._on_disable();
            }
            self._on_in_active();
        }
        self.view_mut()._destroyed = true;
        self._on_destroy();
    }

    /**
     * @internal
     */
    fn _on_awake(&self) {}

    /**
     * @internal
     */
    fn _on_enable(&self) {}

    /**
     * @internal
     */
    fn _on_disable(&self) {}

    /**
     * @internal
     */
    fn _on_destroy(&self) {}

    /**
     * @internal
     */
    fn _on_active(&self) {}

    /**
     * @internal
     */
    fn _on_in_active(&self) {}

    /**
     * @internal
     */
    fn _set_active(&mut self, value: bool) {
        if value {
            if !self.view()._awake {
                self.view_mut()._awake = true;
                self._on_awake();
            }
            // You can do isActive = false in onAwake function.
            if self.view()._entity._is_active_in_hierarchy {
                self._on_active();
                if self.view()._enabled {
                    self._on_enable();
                }
            }
        } else {
            if self.view()._enabled {
                self._on_disable();
            }
            self._on_in_active();
        }
    }
}