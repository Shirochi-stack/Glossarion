#
# CocoaPods spec for the iOS part of flet_glossarion_native. Flutter 3.44
# prefers ios/flet_glossarion_native/Package.swift (Swift Package Manager);
# this spec is used when SwiftPM is disabled. Both compile the same sources.
#
Pod::Spec.new do |s|
  s.name             = 'flet_glossarion_native'
  s.version          = '0.1.0'
  s.summary          = 'Glossarion native services for Flet.'
  s.description      = <<-DESC
Open-in file import, local notifications, beginBackgroundTask and iOS 26
BGContinuedProcessingTask support for the Glossarion Flet app.
                       DESC
  s.homepage         = 'https://github.com/Shirochi-stack/Glossarion'
  s.license          = { :type => 'AGPL-3.0' }
  s.author           = { 'Glossarion contributors' => 'https://github.com/Shirochi-stack/Glossarion' }
  s.source           = { :path => '.' }
  s.source_files     = 'flet_glossarion_native/Sources/flet_glossarion_native/**/*.swift'
  s.dependency 'Flutter'
  s.platform         = :ios, '13.0'
  s.frameworks       = 'BackgroundTasks', 'UserNotifications'

  # Flutter.framework does not contain an i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'
end
