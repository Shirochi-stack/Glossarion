"""
Dialog Animation Utilities
Provides smooth fade-in/fade-out animations for PySide6 dialogs
"""

from PySide6.QtCore import QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import QGraphicsOpacityEffect


def add_fade_in_animation(dialog, duration=250, on_complete=None):
    """
    Add a smooth fade-in animation to a dialog when it's shown.
    
    Args:
        dialog: The QDialog or QWidget to animate
        duration: Animation duration in milliseconds (default: 250ms)
        on_complete: Optional callback function to execute when animation completes
    
    Returns:
        The animation object (keep a reference to prevent garbage collection)
    """
    try:
        # Create opacity effect
        effect = QGraphicsOpacityEffect(dialog)
        dialog.setGraphicsEffect(effect)
        effect.setOpacity(0.0)
        
        # Create fade in animation
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Remove effect when animation finishes to prevent performance issues
        def _cleanup():
            try:
                if dialog and not dialog.isHidden():
                    dialog.setGraphicsEffect(None)
                if on_complete:
                    on_complete()
            except:
                pass
        
        animation.finished.connect(_cleanup)
        
        # Add error handling for the animation property
        try:
            # Verify the effect is still valid before starting
            if effect.parent() is None:
                dialog.setGraphicsEffect(None)
                return None
        except:
            dialog.setGraphicsEffect(None)
            return None
            
        # Start animation
        animation.start()
        
        # Store reference on dialog to prevent garbage collection
        if not hasattr(dialog, '_fade_animations'):
            dialog._fade_animations = []
        dialog._fade_animations.append(animation)
        
        # Add cleanup when dialog is destroyed
        def _stop_animation_on_destroy():
            try:
                if animation and animation.state() == animation.Running:
                    animation.stop()
            except:
                pass
        
        try:
            dialog.destroyed.connect(_stop_animation_on_destroy)
        except:
            pass
        
        return animation
        
    except Exception as e:
        # If animation fails, just show the dialog normally
        try:
            dialog.setGraphicsEffect(None)
        except:
            pass
        return None


def add_fade_out_animation(dialog, duration=200, on_complete=None):
    """
    Add a smooth fade-out animation to a dialog before it closes.
    
    Args:
        dialog: The QDialog or QWidget to animate
        duration: Animation duration in milliseconds (default: 200ms)
        on_complete: Optional callback function to execute when animation completes
                    (typically dialog.hide() or dialog.close())
    
    Returns:
        The animation object (keep a reference to prevent garbage collection)
    """
    try:
        # Create opacity effect
        effect = QGraphicsOpacityEffect(dialog)
        dialog.setGraphicsEffect(effect)
        effect.setOpacity(1.0)
        
        # Create fade out animation
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.InCubic)
        
        # Execute completion callback and cleanup when finished
        def _cleanup():
            try:
                if on_complete:
                    on_complete()
                if dialog and not dialog.isHidden():
                    dialog.setGraphicsEffect(None)
            except:
                pass
        
        animation.finished.connect(_cleanup)
        
        # Add error handling for the animation property
        try:
            # Verify the effect is still valid before starting
            if effect.parent() is None:
                dialog.setGraphicsEffect(None)
                if on_complete:
                    on_complete()
                return None
        except:
            dialog.setGraphicsEffect(None)
            if on_complete:
                on_complete()
            return None
            
        # Start animation
        animation.start()
        
        # Store reference on dialog to prevent garbage collection
        if not hasattr(dialog, '_fade_animations'):
            dialog._fade_animations = []
        dialog._fade_animations.append(animation)
        
        # Add cleanup when dialog is destroyed
        def _stop_animation_on_destroy():
            try:
                if animation and animation.state() == animation.Running:
                    animation.stop()
            except:
                pass
        
        try:
            dialog.destroyed.connect(_stop_animation_on_destroy)
        except:
            pass
        
        return animation
        
    except Exception as e:
        # If animation fails, just execute the completion callback directly
        try:
            dialog.setGraphicsEffect(None)
            if on_complete:
                on_complete()
        except:
            pass
        return None


def show_dialog_with_fade(dialog, duration=250):
    """
    Show a dialog with a fade-in animation.
    Convenience function that combines dialog.show() with fade animation.
    
    Args:
        dialog: The QDialog to show
        duration: Animation duration in milliseconds (default: 250ms)
    """
    # Show dialog first
    dialog.show()
    
    # Then add fade animation
    add_fade_in_animation(dialog, duration)


def exec_dialog_with_fade(dialog, duration=250):
    """
    Execute a modal dialog with a fade-in animation.
    Convenience function that combines dialog.exec() with fade animation.
    
    Args:
        dialog: The QDialog to execute modally
        duration: Animation duration in milliseconds (default: 250ms)
    
    Returns:
        The dialog's exec() result
    """
    # Add fade animation
    add_fade_in_animation(dialog, duration)
    
    # Execute dialog modally
    return dialog.exec()
