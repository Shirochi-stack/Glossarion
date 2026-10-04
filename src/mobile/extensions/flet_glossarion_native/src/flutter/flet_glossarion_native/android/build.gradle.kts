// Android part of the flet_glossarion_native Flutter plugin.
//
// This follows Flutter 3.44's own Kotlin plugin template
// (packages/flutter_tools/templates/plugin/android-kotlin.tmpl/build.gradle.kts.tmpl).
// The Flet 1.0.3 app's root and settings Gradle files are identical to
// Flutter's app template:
//  - AGP 8.11.1 and KGP 2.2.20 are declared in the app's settings.gradle.kts;
//    the buildscript classpath below repeats the same versions.
//  - Kotlin is "built-in". Flutter's Gradle plugin applies kotlin-android to
//    plugin projects that do not declare it, so this script must NOT add it to
//    plugins {} (that is also why the `kotlin {}` accessor exists here).
//  - JDK 17 bytecode, compileSdk 36 and minSdk 24 are Flutter 3.44's defaults.
//    The app itself uses min_sdk_version 26.
//
// Core-library desugaring: the app's root build.gradle.kts makes every
// plugin project evaluationDependsOn(":app"), so :app is fully configured
// (variants, L8 task decision) before this script runs. A plugin cannot
// enable desugaring for it, so notifications are implemented natively here
// instead of through flutter_local_notifications (see README.md).

group = "com.glossarion.flet_glossarion_native"
version = "0.1.0"

buildscript {
    val kotlinVersion = "2.2.20"
    repositories {
        google()
        mavenCentral()
    }

    dependencies {
        classpath("com.android.tools.build:gradle:8.11.1")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlinVersion")
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

plugins {
    id("com.android.library")
}

android {
    namespace = "com.glossarion.flet_glossarion_native"

    compileSdk = 36

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("src/main/kotlin")
        }
    }

    defaultConfig {
        minSdk = 24
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

dependencies {
    // Same androidx.core line as flutter_foreground_task 11.0.3 (core-ktx 1.15.0).
    implementation("androidx.core:core:1.15.0")
}
