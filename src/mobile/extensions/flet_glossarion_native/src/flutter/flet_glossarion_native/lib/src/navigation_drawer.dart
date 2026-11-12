import 'package:flet/flet.dart';
import 'package:flutter/material.dart';

/// The Scaffold bounds the drawer, and SafeArea consumes the current system
/// bars on the client. There is no outer ListView to scroll the footer away.
class GlossarionNavigationDrawerControl extends StatelessWidget {
  final Control control;

  const GlossarionNavigationDrawerControl({super.key, required this.control});

  @override
  Widget build(BuildContext context) {
    return BaseControl(
      control: control,
      child: Drawer(
        width: control.getDouble('width'),
        backgroundColor: control.getColor('bgcolor', context),
        elevation: control.getDouble('elevation'),
        child: SafeArea(
          child: Column(
            children: control.children('controls').map((child) {
              return Expanded(child: ControlWidget(control: child));
            }).toList(),
          ),
        ),
      ),
    );
  }
}
