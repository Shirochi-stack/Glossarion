# halgakos_fab.py
"""
Custom FloatingActionButton with animated spinning Halgakos icon.
KivyMD 1.2.0 compatible.
"""

import os

from kivy.lang import Builder
from kivy.properties import BooleanProperty, StringProperty, NumericProperty
from kivy.animation import Animation
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.graphics import PushMatrix, PopMatrix, Rotate, Color, Ellipse

from kivymd.uix.button import MDFloatingActionButton

# KV for the composite widget
_KV = '''
<HalgakosFAB>:
    size_hint: None, None
    size: dp(56), dp(56)

    # Circular background
    canvas.before:
        Color:
            rgba: app.theme_cls.primary_color if not root.is_active else (0.3, 0.75, 0.3, 1)
        Ellipse:
            pos: self.pos
            size: self.size

    Image:
        id: halgakos_img
        source: root.icon_source
        size_hint: None, None
        size: dp(36), dp(36)
        pos: root.x + (root.width - self.width) / 2, root.y + (root.height - self.height) / 2
        allow_stretch: True
        keep_ratio: True
        canvas.before:
            PushMatrix
            Rotate:
                angle: root.rotation_angle
                origin: self.center
        canvas.after:
            PopMatrix

    # Fallback icon overlay when image not available
    MDIconButton:
        id: fallback_icon
        icon: root.fallback_icon_name
        theme_text_color: "Custom"
        text_color: 1, 1, 1, 1
        pos_hint: {"center_x": 0.5, "center_y": 0.5}
        opacity: 0 if root._has_image else 1
        on_release: root.dispatch('on_release')
'''


class HalgakosFAB(FloatLayout):
    """A FAB that shows a spinning Halgakos icon during translation."""

    is_active = BooleanProperty(False)
    icon_source = StringProperty('')
    fallback_icon_name = StringProperty('play')
    rotation_angle = NumericProperty(0)
    _has_image = BooleanProperty(False)

    def __init__(self, **kwargs):
        self.register_event_type('on_release')
        super().__init__(**kwargs)
        Builder.load_string(_KV)
        self._anim = None
        self._find_icon()

    def _find_icon(self):
        """Locate the Halgakos.png asset."""
        candidates = [
            os.path.join(os.path.dirname(__file__), 'assets', 'Halgakos.png'),
            os.path.join(os.path.dirname(__file__), 'assets', 'Halgakos.ico'),
        ]
        for path in candidates:
            if os.path.isfile(path):
                self.icon_source = path
                self._has_image = True
                return
        self._has_image = False

    def on_release(self, *args):
        """Default handler — override in KV or bind."""
        pass

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            touch.grab(self)
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            if self.collide_point(*touch.pos):
                self.dispatch('on_release')
            return True
        return super().on_touch_up(touch)

    def start_spinning(self):
        """Start the rotation animation."""
        self.is_active = True
        self.fallback_icon_name = 'stop'
        if self._has_image:
            self.stop_spinning_anim()
            self._anim = Animation(rotation_angle=360, duration=1.5)
            self._anim.repeat = True
            self._anim.bind(on_complete=lambda *a: setattr(self, 'rotation_angle', 0))
            self._anim.start(self)

    def show_stopping(self):
        """Show the stopping/clock state."""
        self.fallback_icon_name = 'clock-outline'
        # Slow down the spin
        if self._anim:
            self._anim.cancel(self)
        if self._has_image:
            self._anim = Animation(rotation_angle=self.rotation_angle + 360, duration=4)
            self._anim.repeat = True
            self._anim.start(self)

    def stop_spinning(self):
        """Stop the rotation and reset to play icon."""
        self.is_active = False
        self.fallback_icon_name = 'play'
        self.stop_spinning_anim()
        self.rotation_angle = 0

    def stop_spinning_anim(self):
        if self._anim:
            self._anim.cancel(self)
            self._anim = None
