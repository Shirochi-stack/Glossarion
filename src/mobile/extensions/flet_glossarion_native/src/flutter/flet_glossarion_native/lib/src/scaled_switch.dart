import 'dart:math' as math;

import 'package:flet/flet.dart';
import 'package:flet/src/controls/base_controls.dart';
import 'package:flutter/material.dart';

/// Keeps the label in the normal text layout and sizes only the switch track.
class ScaledSwitchControl extends StatelessWidget {
  final Control control;

  const ScaledSwitchControl({super.key, required this.control});

  @override
  Widget build(BuildContext context) {
    final value = control.getBool('value', false)!;
    final label = control.get('label');
    final theme = Theme.of(context);
    // Glossarion's unscaled bodyMedium is 14 sp. The theme contains the app
    // scale and Flutter's text scaler adds the device accessibility scale.
    final size = theme.textTheme.bodyMedium?.fontSize ?? 14.0;
    final scale = (MediaQuery.textScalerOf(context).scale(size) / 14).clamp(0.5, 1.5).toDouble();
    final switchWidth = 62.0 * scale;
    final switchHeight = 48.0 * scale;
    final ValueChanged<bool>? change = control.disabled
        ? null
        : (next) {
            control.updateProperties({'value': next}, notify: true);
            control.triggerEvent('change', next);
          };

    final toggle = SizedBox(
      width: math.max(48.0, switchWidth),
      height: math.max(48.0, switchHeight),
      child: Center(
        child: SizedBox(
          width: switchWidth,
          height: switchHeight,
          child: FittedBox(child: Switch(value: value, onChanged: change)),
        ),
      ),
    );
    final row = label is String
        ? Row(mainAxisSize: MainAxisSize.min, children: [
            toggle,
            Flexible(child: Text(label, style: theme.textTheme.bodyMedium?.copyWith(
              color: control.disabled ? theme.disabledColor : null,
            ))),
          ])
        : toggle;
    return LayoutControl(
      control: control,
      child: MergeSemantics(
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          excludeFromSemantics: true,
          onTap: change == null ? null : () => change(!value),
          child: row,
        ),
      ),
    );
  }
}
