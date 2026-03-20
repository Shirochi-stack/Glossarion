# android_notification.py
"""
Android notification system for Glossarion.
Uses pyjnius on Android, falls back to plyer on desktop.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Notification channel constants
CHANNEL_ID = 'glossarion_translation'
CHANNEL_NAME = 'Translation Progress'
CHANNEL_DESC = 'Notifications for translation progress and completion'

CHANNEL_COMPLETE_ID = 'glossarion_complete'
CHANNEL_COMPLETE_NAME = 'Translation Complete'
CHANNEL_COMPLETE_DESC = 'Notifications when translation finishes'

NOTIFICATION_PROGRESS_ID = 1001
NOTIFICATION_COMPLETE_ID = 1002


def _is_android():
    try:
        from kivy.utils import platform
        return platform == 'android'
    except ImportError:
        return False


def create_notification_channels():
    """Create notification channels (required for Android 8+).
    
    Must be called once at app startup.
    """
    if not _is_android():
        return

    try:
        from jnius import autoclass

        Context = autoclass('android.content.Context')
        NotificationChannel = autoclass('android.app.NotificationChannel')
        NotificationManager = autoclass('android.app.NotificationManager')
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        
        activity = PythonActivity.mActivity
        nm = activity.getSystemService(Context.NOTIFICATION_SERVICE)
        
        # Build version check
        Build = autoclass('android.os.Build')
        if Build.VERSION.SDK_INT >= 26:
            # Progress channel (low importance — no sound)
            channel = NotificationChannel(
                CHANNEL_ID, CHANNEL_NAME,
                NotificationManager.IMPORTANCE_LOW
            )
            channel.setDescription(CHANNEL_DESC)
            nm.createNotificationChannel(channel)

            # Completion channel (high importance — sound + heads-up)
            channel_done = NotificationChannel(
                CHANNEL_COMPLETE_ID, CHANNEL_COMPLETE_NAME,
                NotificationManager.IMPORTANCE_HIGH
            )
            channel_done.setDescription(CHANNEL_COMPLETE_DESC)
            nm.createNotificationChannel(channel_done)

        logger.info("Notification channels created")
    except Exception as e:
        logger.warning(f"Failed to create notification channels: {e}")


def notify_translation_progress(title, progress, max_progress, message=''):
    """Show/update an ongoing progress notification.
    
    Args:
        title: Notification title (e.g. "Translating novel.epub")
        progress: Current progress value
        max_progress: Maximum progress value
        message: Optional subtitle text (e.g. "Chapter 5/20")
    """
    if _is_android():
        _android_progress_notification(title, progress, max_progress, message)
    else:
        _desktop_progress_notification(title, progress, max_progress, message)


def notify_translation_complete(title, message='Translation finished successfully'):
    """Show a completion notification.
    
    Args:
        title: Notification title
        message: Notification body text
    """
    if _is_android():
        _android_complete_notification(title, message)
    else:
        _desktop_complete_notification(title, message)


def cancel_progress_notification():
    """Cancel the ongoing progress notification."""
    if not _is_android():
        return
    try:
        from jnius import autoclass
        Context = autoclass('android.content.Context')
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        activity = PythonActivity.mActivity
        nm = activity.getSystemService(Context.NOTIFICATION_SERVICE)
        nm.cancel(NOTIFICATION_PROGRESS_ID)
    except Exception as e:
        logger.warning(f"Failed to cancel notification: {e}")


# ---------- Android implementations ----------

def _android_progress_notification(title, progress, max_progress, message):
    try:
        from jnius import autoclass

        Context = autoclass('android.content.Context')
        NotificationCompat = autoclass('androidx.core.app.NotificationCompat')
        NotificationBuilder = autoclass('androidx.core.app.NotificationCompat$Builder')
        PythonActivity = autoclass('org.kivy.android.PythonActivity')

        activity = PythonActivity.mActivity
        nm = activity.getSystemService(Context.NOTIFICATION_SERVICE)

        builder = NotificationBuilder(activity, CHANNEL_ID)
        builder.setContentTitle(title)
        if message:
            builder.setContentText(message)
        builder.setSmallIcon(activity.getApplicationInfo().icon)
        builder.setOngoing(True)  # Can't be swiped away
        builder.setOnlyAlertOnce(True)  # Don't vibrate/sound on update
        builder.setProgress(int(max_progress), int(progress), False)

        # Add percentage text
        if max_progress > 0:
            pct = int(100 * progress / max_progress)
            builder.setSubText(f"{pct}%")

        nm.notify(NOTIFICATION_PROGRESS_ID, builder.build())
    except Exception as e:
        logger.warning(f"Android progress notification failed: {e}")


def _android_complete_notification(title, message):
    try:
        from jnius import autoclass

        Context = autoclass('android.content.Context')
        NotificationCompat = autoclass('androidx.core.app.NotificationCompat')
        NotificationBuilder = autoclass('androidx.core.app.NotificationCompat$Builder')
        PythonActivity = autoclass('org.kivy.android.PythonActivity')

        activity = PythonActivity.mActivity
        nm = activity.getSystemService(Context.NOTIFICATION_SERVICE)

        # Cancel any progress notification first
        nm.cancel(NOTIFICATION_PROGRESS_ID)

        builder = NotificationBuilder(activity, CHANNEL_COMPLETE_ID)
        builder.setContentTitle(title)
        builder.setContentText(message)
        builder.setSmallIcon(activity.getApplicationInfo().icon)
        builder.setAutoCancel(True)  # Dismiss on tap
        builder.setPriority(NotificationCompat.PRIORITY_HIGH)

        # Make it open the app when tapped
        try:
            Intent = autoclass('android.content.Intent')
            PendingIntent = autoclass('android.app.PendingIntent')
            intent = Intent(activity, PythonActivity)
            intent.setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
            pending = PendingIntent.getActivity(
                activity, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
            )
            builder.setContentIntent(pending)
        except Exception:
            pass

        nm.notify(NOTIFICATION_COMPLETE_ID, builder.build())
    except Exception as e:
        logger.warning(f"Android complete notification failed: {e}")


# ---------- Desktop fallbacks (for testing) ----------

def _desktop_progress_notification(title, progress, max_progress, message):
    if max_progress > 0:
        pct = int(100 * progress / max_progress)
        print(f"📱 [Notification] {title}: {pct}% — {message}")


def _desktop_complete_notification(title, message):
    print(f"📱 [Notification] ✅ {title}: {message}")
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name='Glossarion',
            timeout=10,
        )
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"plyer notification failed: {e}")
