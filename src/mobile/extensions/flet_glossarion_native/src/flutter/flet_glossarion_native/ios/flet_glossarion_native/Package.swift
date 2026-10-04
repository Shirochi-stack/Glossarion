// swift-tools-version: 5.9
// Swift Package Manager manifest (Flutter 3.44 integrates iOS plugins through
// SwiftPM by default; flet_glossarion_native.podspec covers CocoaPods builds).
// Layout and the FlutterFramework dependency mirror receive_sharing_intent
// 1.9.0 and flutter_foreground_task 11.0.3.

import PackageDescription

let package = Package(
    name: "flet_glossarion_native",
    platforms: [
        .iOS("13.0")
    ],
    products: [
        .library(name: "flet-glossarion-native", targets: ["flet_glossarion_native"])
    ],
    dependencies: [
        .package(name: "FlutterFramework", path: "../FlutterFramework")
    ],
    targets: [
        .target(
            name: "flet_glossarion_native",
            dependencies: [
                .product(name: "FlutterFramework", package: "FlutterFramework")
            ],
            resources: []
        )
    ]
)
