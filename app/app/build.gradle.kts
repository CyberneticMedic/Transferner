plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.plugin.serialization")
}

android {
    namespace = "io.github.cyberneticmedic.transferner"
    compileSdk {
        version = release(37) { minorApiLevel = 2 }      // Compose BOM 2026.11 needs 37.1+
    }

    defaultConfig {
        applicationId = "io.github.cyberneticmedic.transferner"
        minSdk = 29
        targetSdk = 36
        versionCode = 4
        versionName = "0.3.1"
        ndk { abiFilters += listOf("arm64-v8a") }        // Pixel 7-9 (D-003); keeps the APK small
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")   // prototype: sideloaded, debug-signed
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { compose = true }
    packaging { jniLibs { useLegacyPackaging = false } }
    testOptions { unitTests.isReturnDefaultValues = true }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2026.11.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.core:core-ktx:1.19.1")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.11.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.11.1")
    implementation("androidx.exifinterface:exifinterface:1.4.2")
    implementation("androidx.camera:camera-core:1.7.0")
    implementation("androidx.camera:camera-camera2:1.7.0")
    implementation("androidx.camera:camera-lifecycle:1.7.0")
    implementation("androidx.camera:camera-view:1.7.0")
    implementation("com.google.ai.edge.litert:litert:2.4.0")
    implementation("com.google.ai.edge.litert:litert-gpu:2.4.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.13.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")
    testImplementation("junit:junit:4.13.2")
}

// Unit tests compare the Kotlin pre/post-processing with the Python pipeline's goldens (data, not in git).
tasks.withType<Test>().configureEach {
    systemProperty("transferner.data", System.getenv("TRANSFERNER_DATA") ?: "")
    systemProperty("transferner.server", (project.findProperty("transferner.server") as String?) ?: "")
    systemProperty("java.io.tmpdir", layout.buildDirectory.dir("tmp").get().asFile.path)
    testLogging { showStandardStreams = true }
}
