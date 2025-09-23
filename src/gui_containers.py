import tkinter as tk
from tkinter import ttk
import uuid
import logging

class ResponsiveContainer(tk.Frame):
    """A sophisticated container that adapts to different layouts and themes.
    
    This container supports various layout types (vertical, horizontal, grid) and
    theme variants to create a consistent, modern UI experience.
    """
    
    def __init__(self, master, layout_type='vertical', theme='default', 
                 padding=5, spacing=5, auto_expand=False, **kwargs):
        """Initialize a new ResponsiveContainer.
        
        Args:
            master: Parent widget
            layout_type: Layout strategy ('vertical', 'horizontal', 'grid', 'adaptive')
            theme: Visual theme ('default', 'modern', 'raised', 'sunken')
            padding: Padding around container content
            spacing: Spacing between child elements
            auto_expand: Whether container should automatically expand to fill space
        """
        # Apply theme-specific frame styling
        frame_kwargs = self._get_theme_kwargs(theme)
        frame_kwargs.update(kwargs)  # Allow kwargs to override theme settings
        
        # Initialize the frame with theme styling
        super().__init__(master, **frame_kwargs)
        
        # Container properties
        self.layout_type = layout_type
        self.theme = theme
        self.padding = padding
        self.spacing = spacing
        self.auto_expand = auto_expand
        self.children_widgets = []
        
        # Configure the container to expand properly if requested
        if auto_expand:
            self._setup_auto_expand()
    
    def _get_theme_kwargs(self, theme):
        """Get tkinter styling kwargs based on theme."""
        theme_styles = {
            'default': {
                'bg': '#f0f0f0',
                'highlightthickness': 0,
            },
            'modern': {
                'bg': '#ffffff',
                'highlightthickness': 1,
                'highlightbackground': '#e0e0e0',
            },
            'raised': {
                'bg': '#f5f5f5',
                'relief': 'raised',
                'borderwidth': 1,
            },
            'sunken': {
                'bg': '#fcfcfc',
                'relief': 'sunken',
                'borderwidth': 1,
            }
        }
        
        return theme_styles.get(theme, theme_styles['default'])
    
    def _setup_auto_expand(self):
        """Configure the container to expand within its parent."""
        # Configure grid weights if parent uses grid
        if hasattr(self.master, 'grid_propagate'):
            self.master.grid_propagate(False)
            for i in range(10):  # Configure reasonable number of rows/columns
                self.master.grid_rowconfigure(i, weight=1)
                self.master.grid_columnconfigure(i, weight=1)
        
        # Configure pack propagation
        if hasattr(self.master, 'pack_propagate'):
            self.master.pack_propagate(False)
    
    def add_widget(self, widget, expand=True, fill='both', side='top', sticky='nsew', 
                   row=None, column=None, rowspan=1, columnspan=1, padx=None, pady=None):
        """Add a widget to this container with appropriate layout.
        
        Args:
            widget: Widget to add
            expand: Whether widget should expand to fill space
            fill: Fill direction ('x', 'y', 'both', 'none')
            side: Side for pack layout ('top', 'bottom', 'left', 'right')
            sticky: Grid sticky parameter ('n', 's', 'e', 'w', combinations)
            row, column: Grid position (only for grid layout)
            rowspan, columnspan: Grid span (only for grid layout)
            padx, pady: Padding around widget
        
        Returns:
            widget: The added widget
        """
        # Set default padding based on container spacing if not specified
        padx = padx if padx is not None else self.spacing
        pady = pady if pady is not None else self.spacing
        
        # Track the widget
        self.children_widgets.append({
            'widget': widget,
            'layout_params': {
                'expand': expand,
                'fill': fill,
                'side': side,
                'sticky': sticky,
                'row': row,
                'column': column,
                'rowspan': rowspan,
                'columnspan': columnspan,
                'padx': padx,
                'pady': pady
            }
        })
        
        # Apply the current layout
        self._apply_layout_to_widget(widget, self.children_widgets[-1]['layout_params'])
        
        return widget
    
    def _apply_layout_to_widget(self, widget, params):
        """Apply the appropriate layout manager to widget based on layout_type."""
        # Remove widget from any previous layout
        try:
            widget.pack_forget()
        except:
            pass
        try:
            widget.grid_forget()
        except:
            pass
        
        # Apply new layout
        if self.layout_type == 'vertical' or self.layout_type == 'horizontal':
            side = params['side']
            if self.layout_type == 'vertical':
                side = 'top'
            elif self.layout_type == 'horizontal':
                side = 'left'
            
            widget.pack(
                side=side,
                fill=params['fill'],
                expand=params['expand'],
                padx=params['padx'],
                pady=params['pady']
            )
        
        elif self.layout_type == 'grid':
            # Auto-assign row/column if not specified
            row, col = params['row'], params['column']
            if row is None or col is None:
                row = len(self.children_widgets) // 3  # Assume 3 columns
                col = len(self.children_widgets) % 3
            
            widget.grid(
                row=row,
                column=col,
                rowspan=params['rowspan'],
                columnspan=params['columnspan'],
                sticky=params['sticky'],
                padx=params['padx'],
                pady=params['pady']
            )
    
    def _layout_children(self):
        """Re-layout all children according to current layout_type."""
        # Remove all widgets from current layout
        for child_info in self.children_widgets:
            widget = child_info['widget']
            try:
                widget.pack_forget()
            except:
                pass
            try:
                widget.grid_forget()
            except:
                pass
        
        # Re-add all widgets with current layout
        for child_info in self.children_widgets:
            self._apply_layout_to_widget(child_info['widget'], child_info['layout_params'])
        
        # Update the container
        self.update_idletasks()


class FlexibleDialog(tk.Toplevel):
    """A sophisticated dialog with support for fixed sections, responsive layouts, and themes.
    
    This dialog can have a main scrollable content area plus optional fixed sections at the top/bottom.
    """
    
    def __init__(self, parent, title="Dialog", width=800, height=600, 
                 modal=True, resizable=(True, True), layout='vertical', theme='default'):
        """Create a new flexible dialog window.
        
        Args:
            parent: Parent window
            title: Dialog title
            width, height: Initial dimensions
            modal: Whether dialog is modal (blocks parent)
            resizable: (width_resizable, height_resizable) tuple
            layout: Layout for main container ('vertical', 'horizontal', 'grid')
            theme: Visual theme ('default', 'modern', 'raised', 'sunken')
        """
        super().__init__(parent)
        self.title(title)
        
        # Configure dialog appearance
        self.geometry(f"{width}x{height}")
        self.minsize(400, 300)
        self.resizable(resizable[0], resizable[1])
        
        # Make it modal if requested
        if modal:
            self.transient(parent)
            self.grab_set()
        
        # Configure the dialog theme
        self._apply_theme(theme)
        
        # Main sections
        self.top_frame = tk.Frame(self, **self._get_section_style('top', theme))
        self.main_container = tk.Frame(self, **self._get_section_style('main', theme))
        self.bottom_frame = tk.Frame(self, **self._get_section_style('bottom', theme))
        
        # Setup basic layout with grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Main container should expand
        
        # Place frames in grid
        self.top_frame.grid(row=0, column=0, sticky='ew')
        self.main_container.grid(row=1, column=0, sticky='nsew')
        self.bottom_frame.grid(row=2, column=0, sticky='ew')
        
        # Grid configuration for sections
        self.top_frame.grid_columnconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        
        # Initialize sections dictionary
        self.sections = {}
        self.section_weights = {}
    
    def _apply_theme(self, theme):
        """Apply theme styling to the dialog."""
        # Base styling
        self.configure(**self._get_theme_kwargs(theme))
        
        # For modern themes, add special styling
        if theme == 'modern':
            style = ttk.Style()
            style.configure('Flexible.TFrame', background='#ffffff')
            style.configure('Flexible.TButton', background='#f0f0f0', padding=5)
    
    def _get_theme_kwargs(self, theme):
        """Get tkinter styling kwargs based on theme."""
        theme_styles = {
            'default': {
                'bg': '#f0f0f0',
            },
            'modern': {
                'bg': '#ffffff',
            },
            'raised': {
                'bg': '#f5f5f5',
            },
            'sunken': {
                'bg': '#fcfcfc',
            }
        }
        
        return theme_styles.get(theme, theme_styles['default'])
    
    def _get_section_style(self, section_type, theme):
        """Get styling for different dialog sections."""
        base_style = self._get_theme_kwargs(theme)
        
        section_styles = {
            'top': {
                'relief': 'flat',
            },
            'main': {
                'relief': 'flat',
            },
            'bottom': {
                'relief': 'flat',
            }
        }
        
        # Combine base style with section-specific style
        style = base_style.copy()
        style.update(section_styles.get(section_type, {}))
        
        return style
    
    def add_section(self, name, container, position='top', weight=0, fixed=True):
        """Add a named section to the dialog.
        
        Args:
            name: Section identifier
            container: Widget to add as a section
            position: Where to add ('top', 'main', 'bottom')
            weight: Grid weight (0 = fixed size, >0 = proportional)
            fixed: Whether the section is fixed (non-scrollable)
            
        Returns:
            container: The added container
        """
        self.sections[name] = container
        self.section_weights[name] = weight
        
        # Determine target frame based on position
        target_frame = {
            'top': self.top_frame,
            'main': self.main_container,
            'bottom': self.bottom_frame
        }.get(position, self.main_container)
        
        # Configure target frame grid
        next_row = len([s for s in self.sections.values() 
                      if s.master == target_frame])
        
        target_frame.grid_rowconfigure(next_row, weight=weight)
        
        # Add container to target frame
        container.grid(row=next_row, column=0, sticky='nsew', padx=5, pady=5)
        
        return container


class SectionContainer(tk.Frame):
    """A container for a specific section of the UI with a title and content area."""
    
    def __init__(self, master, title=None, collapsible=False, theme='default', **kwargs):
        """Create a new section container.
        
        Args:
            master: Parent widget
            title: Section title (None for no header)
            collapsible: Whether section can be collapsed
            theme: Visual theme
        """
        # Apply theme styling
        frame_kwargs = self._get_theme_kwargs(theme)
        frame_kwargs.update(kwargs)
        
        super().__init__(master, **frame_kwargs)
        
        # Setup container styling
        self._id = str(uuid.uuid4())
        self.theme = theme
        self.collapsible = collapsible
        self.collapsed = False
        
        # Create header if title is provided
        self.header_frame = None
        if title:
            self.header_frame = tk.Frame(self)
            self.header_frame.pack(fill='x', expand=False)
            
            # Title label with theme styling
            header_style = self._get_header_style(theme)
            self.title_label = tk.Label(self.header_frame, text=title, **header_style)
            self.title_label.pack(side='left', padx=5, pady=5)
            
            # Add collapse button if requested
            if collapsible:
                self.toggle_btn = tk.Button(
                    self.header_frame, 
                    text="▼", 
                    command=self.toggle_collapse,
                    relief='flat',
                    **self._get_button_style(theme)
                )
                self.toggle_btn.pack(side='right', padx=5, pady=5)
        
        # Content area
        self.content_frame = tk.Frame(self, **self._get_content_style(theme))
        self.content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configure grid layout for content
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
    
    def _get_theme_kwargs(self, theme):
        """Get tkinter styling kwargs based on theme."""
        theme_styles = {
            'default': {
                'bg': '#f0f0f0',
                'highlightthickness': 0,
            },
            'modern': {
                'bg': '#ffffff',
                'highlightthickness': 1,
                'highlightbackground': '#e0e0e0',
            },
            'raised': {
                'bg': '#f5f5f5',
                'relief': 'raised',
                'borderwidth': 1,
            },
            'sunken': {
                'bg': '#fcfcfc',
                'relief': 'sunken',
                'borderwidth': 1,
            }
        }
        
        return theme_styles.get(theme, theme_styles['default'])
    
    def _get_header_style(self, theme):
        """Get header styling based on theme."""
        header_styles = {
            'default': {
                'bg': '#e6e6e6',
                'fg': '#333333',
                'font': ('Arial', 10, 'bold')
            },
            'modern': {
                'bg': '#f8f8f8',
                'fg': '#444444',
                'font': ('Arial', 10, 'bold')
            },
            'raised': {
                'bg': '#ececec',
                'fg': '#333333',
                'font': ('Arial', 10, 'bold')
            },
            'sunken': {
                'bg': '#f8f8f8',
                'fg': '#444444',
                'font': ('Arial', 10, 'bold')
            }
        }
        
        return header_styles.get(theme, header_styles['default'])
    
    def _get_button_style(self, theme):
        """Get button styling based on theme."""
        button_styles = {
            'default': {
                'bg': '#e6e6e6',
                'fg': '#333333',
            },
            'modern': {
                'bg': '#f8f8f8',
                'fg': '#444444',
            },
            'raised': {
                'bg': '#ececec',
                'fg': '#333333',
            },
            'sunken': {
                'bg': '#f8f8f8',
                'fg': '#444444',
            }
        }
        
        return button_styles.get(theme, button_styles['default'])
    
    def _get_content_style(self, theme):
        """Get content area styling based on theme."""
        content_styles = {
            'default': {
                'bg': '#f9f9f9',
            },
            'modern': {
                'bg': '#ffffff',
            },
            'raised': {
                'bg': '#f9f9f9',
            },
            'sunken': {
                'bg': '#ffffff',
            }
        }
        
        return content_styles.get(theme, content_styles['default'])
    
    def toggle_collapse(self):
        """Toggle the collapsed state of the section."""
        if not self.collapsible:
            return
            
        self.collapsed = not self.collapsed
        
        if self.collapsed:
            self.content_frame.pack_forget()
            self.toggle_btn.config(text="▶")
        else:
            self.content_frame.pack(fill='both', expand=True)
            self.toggle_btn.config(text="▼")
    
    def add_widget(self, widget, expand=True, fill='both', side='top', sticky='nsew', 
                  row=None, column=None, rowspan=1, columnspan=1, padx=5, pady=5):
        """Add a widget to this section's content area.
        
        Supports both pack and grid geometry managers.
        """
        if row is not None and column is not None:
            # Use grid layout
            widget.grid(
                row=row, 
                column=column, 
                rowspan=rowspan, 
                columnspan=columnspan, 
                sticky=sticky,
                padx=padx,
                pady=pady
            )
        else:
            # Use pack layout
            widget.pack(
                fill=fill, 
                expand=expand, 
                side=side,
                padx=padx,
                pady=pady
            )
        
        return widget