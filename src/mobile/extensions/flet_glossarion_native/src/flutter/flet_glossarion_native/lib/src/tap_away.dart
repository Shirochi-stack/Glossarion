import 'package:flet/flet.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/widgets.dart';

/// "GlossarionTapAway": its `content`, and a `tap_away` event when the user taps
/// the page outside it (a dropdown that closes when tapped away from). Flet 1.0.3
/// exposes Flutter's [TapRegion] only as a text field's `on_tap_outside`. Off
/// while `enabled` is false, so a closed dropdown costs a tap nothing.
///
/// A tap, not a touch: a finger that moves past the touch slop is a scroll, which
/// leaves the dropdown open. The page, not the screen: an open dropdown menu, a
/// text selection toolbar, a sheet or a dialog (even one opened from inside
/// `content`) sits outside the page's subtree, so using one leaves it open too.
class GlossarionTapAwayControl extends StatefulWidget {
  final Control control;

  GlossarionTapAwayControl({Key? key, required this.control})
      : super(key: key ?? ValueKey("control_${control.id}"));

  @override
  State<GlossarionTapAwayControl> createState() => _GlossarionTapAwayState();
}

class _GlossarionTapAwayState extends State<GlossarionTapAwayControl> {
  ModalRoute<Object?>? _route;
  int? _pointer;
  Offset _origin = Offset.zero;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _route = ModalRoute.of(context);
  }

  @override
  void dispose() {
    _stopFollowing();
    super.dispose();
  }

  bool _onThisPage(PointerDownEvent down) {
    final RenderObject? page = _route?.subtreeContext?.findRenderObject();
    if (page == null) {
      return true;
    }
    final HitTestResult hit = HitTestResult();
    WidgetsBinding.instance.hitTestInView(hit, down.position, down.viewId);
    return hit.path.any((HitTestEntry entry) => identical(entry.target, page));
  }

  void _onTapOutside(PointerDownEvent down) {
    if (!_onThisPage(down)) {
      return;
    }
    _stopFollowing();
    _pointer = down.pointer;
    _origin = down.position;
    GestureBinding.instance.pointerRouter.addRoute(down.pointer, _onPointer);
  }

  void _onPointer(PointerEvent event) {
    if (event is PointerMoveEvent) {
      if ((event.position - _origin).distance <= kTouchSlop) {
        return;
      }
    } else if (event is! PointerUpEvent && event is! PointerCancelEvent) {
      return;
    }
    _stopFollowing();
    if (event is PointerUpEvent) {
      widget.control.triggerEvent("tap_away");
    }
  }

  void _stopFollowing() {
    final int? pointer = _pointer;
    if (pointer != null) {
      GestureBinding.instance.pointerRouter.removeRoute(pointer, _onPointer);
      _pointer = null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return LayoutControl(
      control: widget.control,
      child: TapRegion(
        enabled: widget.control.getBool("enabled", true)!,
        onTapOutside: _onTapOutside,
        child: widget.control.buildWidget("content") ?? const SizedBox.shrink(),
      ),
    );
  }
}
