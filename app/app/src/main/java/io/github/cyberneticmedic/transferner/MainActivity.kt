package io.github.cyberneticmedic.transferner

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import io.github.cyberneticmedic.transferner.ui.AppViewModel
import io.github.cyberneticmedic.transferner.ui.ForageScreen
import io.github.cyberneticmedic.transferner.ui.LiveScreen
import io.github.cyberneticmedic.transferner.ui.LogScreen
import io.github.cyberneticmedic.transferner.ui.ResultScreen
import io.github.cyberneticmedic.transferner.ui.Screen
import io.github.cyberneticmedic.transferner.ui.SettingsScreen
import io.github.cyberneticmedic.transferner.ui.SetupScreen
import io.github.cyberneticmedic.transferner.ui.SpeciesScreen
import io.github.cyberneticmedic.transferner.ui.TransfernerTheme

class MainActivity : ComponentActivity() {
    private val vm: AppViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { TransfernerTheme { App(vm) { finish() } } }
    }
}

@Composable
private fun App(vm: AppViewModel, exit: () -> Unit) {
    val screen by vm.screen.collectAsState()
    val message by vm.message.collectAsState()
    val snack = remember { SnackbarHostState() }
    LaunchedEffect(message) {
        message?.let { snack.showSnackbar(it); vm.message.value = null }
    }
    BackHandler { if (!vm.back()) exit() }
    Box(Modifier.fillMaxSize()) {
        when (screen) {
            Screen.SETUP -> SetupScreen(vm)
            Screen.LIVE -> LiveScreen(vm)
            Screen.RESULT -> ResultScreen(vm)
            Screen.SPECIES -> SpeciesScreen(vm)
            Screen.LOG -> LogScreen(vm)
            Screen.SETTINGS -> SettingsScreen(vm)
            Screen.FORAGE -> ForageScreen(vm)
        }
        SnackbarHost(snack, Modifier.align(Alignment.BottomCenter).navigationBarsPadding().padding(bottom = 180.dp))
    }
}
