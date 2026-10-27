import 'package:flet/flet.dart';
import 'package:flutter/material.dart';

/// "GlossarionDraggableSheet" (hotfix 5): Flutter's [DraggableScrollableSheet]
/// as the content of a Flet BottomSheet (`scrollable=True`), for the composer's
/// mini model picker. Flet 1.0.3 has no draggable sheet, and a Python-driven
/// height would follow the finger one round trip late.
///
/// The sheet opens at `initial_size` of the height the BottomSheet gives it. A
/// swipe up on the rows grows it to `max_size` before the rows scroll; a swipe
/// down shrinks it, snaps back to `initial_size` when let go above it, and
/// closes the BottomSheet at `min_size` (`shouldCloseOnMinExtent`). `header`
/// scrolls with the rows (so it drags the sheet too); `footer` stays pinned
/// under them.
class GlossarionDraggableSheetControl extends StatelessWidget {
  final Control control;

  GlossarionDraggableSheetControl({Key? key, required this.control})
      : super(key: key ?? ValueKey("control_${control.id}"));

  @override
  Widget build(BuildContext context) {
    final double maxSize =
        control.getDouble("max_size", 1.0)!.clamp(0.05, 1.0).toDouble();
    final double minSize =
        control.getDouble("min_size", 0.25)!.clamp(0.0, maxSize).toDouble();
    final double initial = control
        .getDouble("initial_size", 0.5)!
        .clamp(minSize, maxSize)
        .toDouble();
    final bool snap = control.getBool("snap", true)!;
    final Widget? header = control.buildWidget("header");
    final Widget? footer = control.buildWidget("footer");
    final List<Control> items = control.children("controls");
    final int offset = header != null ? 1 : 0;

    final Widget sheet = DraggableScrollableSheet(
      expand: false,
      initialChildSize: initial,
      minChildSize: minSize,
      maxChildSize: maxSize,
      snap: snap,
      // snap sizes must lie strictly between the min and max sizes
      snapSizes: snap && initial > minSize && initial < maxSize
          ? <double>[initial]
          : null,
      shouldCloseOnMinExtent: true,
      builder: (BuildContext context, ScrollController scrollController) {
        final Widget list = ListView.builder(
          controller: scrollController,
          // a short list still drags the sheet
          physics: const AlwaysScrollableScrollPhysics(),
          padding: EdgeInsets.zero,
          itemCount: items.length + offset,
          itemBuilder: (BuildContext context, int index) {
            if (index < offset) {
              return header!;
            }
            final Control item = items[index - offset];
            return ControlWidget(key: ValueKey(item.id), control: item);
          },
        );
        if (footer == null) {
          return list;
        }
        return Column(
          children: <Widget>[Expanded(child: list), footer],
        );
      },
    );
    return LayoutControl(control: control, child: sheet);
  }
}
