import 'package:flet/flet.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';

import 'draggable_sheet.dart';
import 'native_service.dart';

class Extension extends FletExtension {
  @override
  void ensureInitialized() {
    // flutter_foreground_task delivers TaskHandler data to the main isolate
    // through a named port that must be registered before the service starts
    // (its README asks for this in main(); Flet calls ensureInitialized() from
    // main() before runApp()).
    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
      FlutterForegroundTask.initCommunicationPort();
    }
  }

  @override
  Widget? createWidget(Key? key, Control control) {
    switch (control.type) {
      case "GlossarionDraggableSheet":
        return GlossarionDraggableSheetControl(key: key, control: control);
      default:
        return null;
    }
  }

  @override
  FletService? createService(Control control) {
    switch (control.type) {
      case "GlossarionNative":
        return GlossarionNativeService(control: control);
      default:
        return null;
    }
  }
}
