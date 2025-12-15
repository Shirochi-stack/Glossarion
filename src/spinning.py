# spinning.py
"""
Spinning Icon Helper Module for Glossarion
Provides reusable spinning animation functionality for icons throughout the application.
"""

import os
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QTransform


class SpinningIcon:
    """Helper class for creating and animating spinning icons"""
    
    @staticmethod
    def animate_icon(icon_label):
        """
        Animate an icon with a spin rotation
        
        Args:
            icon_label (QLabel): The QLabel containing the icon to animate
        """
        if not icon_label:
            return
        
        # Animation guard: prevent multiple simultaneous animations
        if hasattr(icon_label, '_is_animating') and icon_label._is_animating:
            return
        
        # Stop any existing timer
        if hasattr(icon_label, '_rotation_timer'):
            icon_label._rotation_timer.stop()
            icon_label._rotation_timer.deleteLater()
            delattr(icon_label, '_rotation_timer')
        
        # Get current pixmap for rotation
        current_pixmap = icon_label.pixmap()
        if current_pixmap and not current_pixmap.isNull():
            # Set animation guard
            icon_label._is_animating = True
            
            # Store original pixmap if not already stored
            if not hasattr(icon_label, '_original_pixmap'):
                icon_label._original_pixmap = current_pixmap.copy()
            
            original_pixmap = icon_label._original_pixmap
            
            # Animate through rotation
            def rotate_frame(angle):
                if not icon_label._is_animating:  # Check if animation was cancelled
                    return
                    
                transform = QTransform()
                transform.rotate(angle)
                rotated_pixmap = original_pixmap.transformed(transform, Qt.SmoothTransformation)
                
                # Ensure the rotated pixmap fits in the label
                if not rotated_pixmap.isNull():
                    # Get original size from the icon label
                    label_size = icon_label.size()
                    scaled_rotated = rotated_pixmap.scaled(
                        label_size.width(), label_size.height(), 
                        Qt.KeepAspectRatio, 
                        Qt.SmoothTransformation
                    )
                    icon_label.setPixmap(scaled_rotated)
            
            # Create timer-based rotation animation
            rotation_timer = QTimer()
            icon_label._rotation_timer = rotation_timer  # Store reference for cleanup
            rotation_steps = 36  # 10-degree steps for smooth animation
            rotation_step = 0
            
            def animate_rotation():
                nonlocal rotation_step
                if rotation_step < rotation_steps and icon_label._is_animating:
                    angle = (rotation_step * 360) // rotation_steps
                    rotate_frame(angle)
                    rotation_step += 1
                else:
                    # Animation complete, restore original and cleanup
                    if hasattr(icon_label, '_original_pixmap'):
                        icon_label.setPixmap(icon_label._original_pixmap)
                    icon_label._is_animating = False
                    rotation_timer.stop()
                    if hasattr(icon_label, '_rotation_timer'):
                        delattr(icon_label, '_rotation_timer')
            
            rotation_timer.timeout.connect(animate_rotation)
            rotation_timer.start(14)  # ~14ms per frame for smooth 500ms total
        else:
            # Fallback for text-based icons (like gear emoji)
            # Animation guard for text icons too
            if hasattr(icon_label, '_is_pulse_animating') and icon_label._is_pulse_animating:
                return
            
            icon_label._is_pulse_animating = True
            original_stylesheet = icon_label.styleSheet()
            
            def pulse_effect():
                if not hasattr(icon_label, '_is_pulse_animating') or not icon_label._is_pulse_animating:
                    return
                    
                # Quick scale pulse effect for text icons
                icon_label.setStyleSheet(original_stylesheet + """
                    QLabel { 
                        font-size: 18px; 
                        color: #7fb3d4; 
                    }
                """)
                
                # Return to normal after short delay
                def restore_style():
                    if hasattr(icon_label, '_is_pulse_animating'):
                        icon_label.setStyleSheet(original_stylesheet)
                        icon_label._is_pulse_animating = False
                
                QTimer.singleShot(150, restore_style)
            
            pulse_effect()
    
    @staticmethod
    def create_icon_label(size=24, base_dir=None):
        """
        Create a Halgakos.ico icon label with fallback
        
        Args:
            size (int): Size of the icon in pixels (default: 24)
            base_dir (str): Base directory to look for the icon (optional)
            
        Returns:
            QLabel: Icon label ready to be added to layouts
        """
        icon_label = QLabel()
        icon_loaded = False
        
        # Try multiple possible paths for the icon
        possible_paths = []
        
        # Method 1: Use provided base_dir
        if base_dir and os.path.isdir(base_dir):
            possible_paths.append(os.path.join(base_dir, 'Halgakos.ico'))
        
        # Method 2: Use current working directory
        possible_paths.append(os.path.join(os.getcwd(), 'Halgakos.ico'))
        
        # Method 3: Use script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths.append(os.path.join(script_dir, 'Halgakos.ico'))
        
        # Method 4: Check parent directory (common for project structure)
        parent_dir = os.path.dirname(script_dir)
        possible_paths.append(os.path.join(parent_dir, 'Halgakos.ico'))
        
        try:
            for ico_path in possible_paths:
                if os.path.isfile(ico_path):
                    # Load icon as pixmap with high quality
                    pixmap = QPixmap(ico_path)
                    if not pixmap.isNull():
                        # Scale to appropriate size
                        scaled_pixmap = pixmap.scaled(
                            size, size, 
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                        icon_label.setPixmap(scaled_pixmap)
                        icon_label.setFixedSize(size, size)
                        icon_label.setAlignment(Qt.AlignCenter)
                        # Transparent background
                        icon_label.setStyleSheet("""
                            QLabel {
                                background: transparent;
                                border: none;
                            }
                        """)
                        icon_loaded = True
                        break
        except Exception:
            pass
        
        # If no icon was loaded, create a simple fallback
        if not icon_loaded:
            icon_label.setText("⚙️")  # Gear emoji as fallback
            icon_label.setFixedSize(size, size)
            icon_label.setAlignment(Qt.AlignCenter)
            icon_label.setStyleSheet("""
                QLabel {
                    background: transparent;
                    border: none;
                    font-size: 14px;
                    color: #5a9fd4;
                }
            """)
        
        return icon_label
    
    @staticmethod
    def create_spinning_checkbox_with_icon(checkbox, icon_size=20, base_dir=None):
        """
        Create a spinning icon and connect it to a checkbox toggle
        
        Args:
            checkbox (QCheckBox): The checkbox to connect the animation to
            icon_size (int): Size of the icon in pixels (default: 20)
            base_dir (str): Base directory to look for the icon (optional)
            
        Returns:
            QLabel: The icon label that will spin when checkbox is toggled
        """
        icon_label = SpinningIcon.create_icon_label(icon_size, base_dir)
        checkbox.toggled.connect(lambda: SpinningIcon.animate_icon(icon_label))
        return icon_label


# Convenience functions for easy importing
def animate_icon(icon_label):
    """Animate an icon with a spin rotation"""
    return SpinningIcon.animate_icon(icon_label)


def create_icon_label(size=24, base_dir=None):
    """Create a Halgakos.ico icon label with fallback"""
    return SpinningIcon.create_icon_label(size, base_dir)


def create_spinning_checkbox_with_icon(checkbox, icon_size=20, base_dir=None):
    """Create a spinning icon and connect it to a checkbox toggle"""
    return SpinningIcon.create_spinning_checkbox_with_icon(checkbox, icon_size, base_dir)
