import datetime
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import logging

from config import config
from database import PotholeDatabase
from models import Pothole

logger = logging.getLogger(__name__)

class PotholeBot:
    def __init__(self, db: PotholeDatabase):
        self.db = db
        self.application = Application.builder().token(config.BOT_TOKEN).build()
        self.setup_handlers()

    def setup_handlers(self):
        """Set up bot command handlers"""
        self.application.add_handler(CommandHandler('start', self.start))
        self.application.add_handler(CommandHandler('help', self.help_command_interactive))
        self.application.add_handler(CommandHandler('locations', self.display_locations))
        self.application.add_handler(CommandHandler('map', self.send_map))
        self.application.add_handler(CommandHandler('stats', self.send_stats))

        # Region and location handlers
        self.application.add_handler(CallbackQueryHandler(self.show_locations_in_region, pattern='^region:'))
        self.application.add_handler(CallbackQueryHandler(self.show_region_stats, pattern='^stats:'))
        self.application.add_handler(CallbackQueryHandler(self.back_to_regions, pattern='^back_to_regions$'))
        self.application.add_handler(CallbackQueryHandler(self.noop_handler, pattern='^noop$'))

        # Severity handlers
        self.application.add_handler(CallbackQueryHandler(self.show_potholes_by_severity, pattern='^severity:'))

        # Help handlers
        self.application.add_handler(CallbackQueryHandler(self.help_topic_handler, pattern='^help:(?!menu)'))
        self.application.add_handler(CallbackQueryHandler(self.help_menu_handler, pattern='^help:menu$'))

        # Location coordinate handler (must be last to avoid conflicts)
        self.application.add_handler(CallbackQueryHandler(self.send_location, pattern=r'^-?\d+\.\d+,-?\d+\.\d+$'))

        self.application.add_handler(CommandHandler('severity', self.display_by_severity))
        self.application.add_handler(CommandHandler('export', self.export_csv))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send welcome message"""
        message = (
            "*🚗 Pothole Info Bot 🚗*\n\n"
            "Welcome! This bot helps you track detected potholes.\n\n"
            "*Available Commands:*\n"
            "/start - Show this help message\n"
            "/locations - Browse pothole locations by region\n"
            "/map - Get a Google Maps link with all locations\n"
            "/stats - View detection statistics\n\n"
            "Stay safe on the roads! 🛣️"
        )
        await update.message.reply_text(message, parse_mode='Markdown')

    async def send_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send detection statistics"""
        stats = self.db.get_statistics()
        total = stats['total']
        severity_stats = stats['by_severity']
        top_regions = stats['top_regions']

        message = f"*📊 Pothole Detection Statistics*\n\nTotal Potholes Detected: {total}\n"
        for severity, count in severity_stats.items():
            message += f"{severity.capitalize()}: {count}\n"

        message += "\n*Top Regions:*\n"
        for region, count in top_regions:
            message += f"• {region}: {count} potholes\n"

        await update.message.reply_text(message, parse_mode='Markdown')

    async def send_map(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send Google Maps link with all pothole locations"""
        potholes = self.db.get_potholes()
        if not potholes:
            await update.message.reply_text("No pothole locations available.")
            return

        base_url = "https://www.google.com/maps/dir/?api=1"
        locations = [f"{p.latitude},{p.longitude}" for p in potholes]
        destination = locations[0]
        waypoints = "|".join(locations[1:])

        url = f"{base_url}&destination={destination}"
        if waypoints:
            from urllib.parse import quote
            url += f"&waypoints={quote(waypoints)}"

        message = f"📍 *Pothole Locations Map*\n\nTotal locations: {len(potholes)}\n"
        message += f"[View on Google Maps]({url})"
        await update.message.reply_text(message, parse_mode='Markdown')

    async def display_locations(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Display regions with pothole detections"""
        # Get all potholes
        potholes = self.db.get_potholes()

        if not potholes:
            await update.message.reply_text("No pothole locations available.")
            return

        # Extract unique regions and count potholes per region
        region_counts = {}
        for pothole in potholes:
            if pothole.region:
                region_counts[pothole.region] = region_counts.get(pothole.region, 0) + 1

        if not region_counts:
            await update.message.reply_text("No regions with pothole data available.")
            return

        keyboard = []
        for region in sorted(region_counts.keys()):
            count = region_counts[region]
            button_text = f"{region} ({count} potholes)"
            callback_data = f"region:{region}:all:0"  # region:name:filter:page
            keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Select a region to view pothole locations:",
            reply_markup=reply_markup
        )

    async def show_locations_in_region(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show locations in selected region with sorting options"""
        query = update.callback_query
        await query.answer()

        # Parse callback data
        data_parts = query.data.split(":")
        region = data_parts[1]
        sort_by = data_parts[2] if len(data_parts) > 2 else "all"
        page = int(data_parts[3]) if len(data_parts) > 3 else 0

        ITEMS_PER_PAGE = 5

        # Get potholes for the region
        if sort_by == "all":
            potholes = self.db.get_potholes(filters={'region': region}, sort_by='timestamp', sort_order='DESC')
        else:
            # Filter by both region and severity
            potholes = self.db.get_potholes(
                filters={'region': region, 'severity': sort_by},
                sort_by='depth',
                sort_order='DESC'
            )

        if not potholes:
            await query.message.edit_text(f"No potholes found in {region}" +
                                          (f" with {sort_by} severity." if sort_by != "all" else "."))
            return

        # Calculate pagination
        total_pages = (len(potholes) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        start_idx = page * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, len(potholes))

        # Build message
        severity_emojis = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }

        title = f"*Potholes in {region}*"
        if sort_by != "all":
            title += f" - {sort_by.capitalize()} Severity"

        message = f"{title}\n"
        message += f"_Page {page + 1} of {total_pages} • Total: {len(potholes)} potholes_\n\n"

        # Display potholes with proper data type handling
        for i, pothole in enumerate(potholes[start_idx:end_idx], start=start_idx + 1):
            emoji = severity_emojis.get(pothole.severity.value, '⚪')
            message += f"{i}. {emoji} *{pothole.city}*\n"
            message += f"   Severity: {pothole.severity.value.capitalize()}\n"

            # Handle depth and area with proper type conversion
            try:
                if isinstance(pothole.depth, bytes):
                    # If it's bytes, try to decode and convert
                    depth_str = pothole.depth.decode('utf-8', errors='ignore')
                    depth = float(depth_str) if depth_str.replace('.', '').replace('-', '').isdigit() else 0.0
                else:
                    depth = float(pothole.depth) if pothole.depth is not None else 0.0

                # Convert depth from meters to centimeters
                depth_cm = depth * 100

                if isinstance(pothole.area, bytes):
                    # If it's bytes, try to decode and convert
                    area_str = pothole.area.decode('utf-8', errors='ignore')
                    area = float(area_str) if area_str.replace('.', '').replace('-', '').isdigit() else 0.0
                else:
                    area = float(pothole.area) if pothole.area is not None else 0.0

                message += f"   📏 Depth: {depth_cm:.1f}cm | 📐 Area: {area:.0f}px\n"
            except (ValueError, AttributeError, UnicodeDecodeError) as e:
                logger.error(f"Error converting depth/area for pothole {i}: {e}")
                message += f"   📏 Depth: N/A | 📐 Area: N/A\n"

            message += f"   📍 `{pothole.latitude:.4f}, {pothole.longitude:.4f}`\n\n"

        # Rest of the method remains the same...
        # Build keyboard
        keyboard = []

        # Severity filter buttons (first row)
        severity_buttons = []
        if sort_by != "all":
            severity_buttons.append(InlineKeyboardButton("📊 Show All", callback_data=f"region:{region}:all:0"))
        if sort_by != "low":
            severity_buttons.append(InlineKeyboardButton("🟢 Low", callback_data=f"region:{region}:low:0"))
        if sort_by != "medium":
            severity_buttons.append(InlineKeyboardButton("🟡 Medium", callback_data=f"region:{region}:medium:0"))

        if severity_buttons:
            keyboard.append(severity_buttons)

        # More severity buttons (second row)
        severity_buttons2 = []
        if sort_by != "high":
            severity_buttons2.append(InlineKeyboardButton("🟠 High", callback_data=f"region:{region}:high:0"))
        if sort_by != "critical":
            severity_buttons2.append(InlineKeyboardButton("🔴 Critical", callback_data=f"region:{region}:critical:0"))

        if severity_buttons2:
            keyboard.append(severity_buttons2)

        # Navigation buttons
        if total_pages > 1:
            nav_buttons = []
            if page > 0:
                nav_buttons.append(InlineKeyboardButton("⬅️ Previous",
                                                        callback_data=f"region:{region}:{sort_by}:{page - 1}"))
            nav_buttons.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop"))
            if page < total_pages - 1:
                nav_buttons.append(InlineKeyboardButton("Next ➡️",
                                                        callback_data=f"region:{region}:{sort_by}:{page + 1}"))
            keyboard.append(nav_buttons)

        # Add location buttons for current page
        for pothole in potholes[start_idx:end_idx]:
            location_text = f"📍 View on map: {pothole.city}"
            location_data = f"{pothole.latitude},{pothole.longitude}"
            keyboard.append([InlineKeyboardButton(location_text, callback_data=location_data)])

        # Statistics button
        keyboard.append([InlineKeyboardButton("📊 Region Statistics", callback_data=f"stats:{region}")])

        # Back button
        keyboard.append([InlineKeyboardButton("🔙 Back to Regions", callback_data="back_to_regions")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def show_potholes_by_severity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show potholes filtered by severity with pagination"""
        query = update.callback_query
        await query.answer()

        # Parse callback data
        data_parts = query.data.split(":")
        severity = data_parts[1]
        page = int(data_parts[2]) if len(data_parts) > 2 else 0

        ITEMS_PER_PAGE = 5

        # Get potholes based on severity
        if severity == "all":
            potholes = self.db.get_potholes(sort_by='severity', sort_order='DESC')
            title = "All Potholes (sorted by severity)"
        else:
            potholes = self.db.get_potholes(filters={'severity': severity}, sort_by='depth', sort_order='DESC')
            title = f"{severity.capitalize()} Severity Potholes"

        if not potholes:
            await query.message.edit_text(f"No {severity} severity potholes found.")
            return

        # Calculate pagination
        total_pages = (len(potholes) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        start_idx = page * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, len(potholes))

        # Build message
        severity_emojis = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }

        message = f"*{title}*\n"
        message += f"_Page {page + 1} of {total_pages} • Total: {len(potholes)} potholes_\n\n"

        for i, p in enumerate(potholes[start_idx:end_idx], start=start_idx + 1):
            emoji = severity_emojis.get(p.severity.value, '⚪')
            message += f"{i}. {emoji} *{p.severity.value.upper()}* - {p.city}, {p.region}\n"

            # Handle depth and area with proper type conversion
            try:
                if isinstance(p.depth, bytes):
                    depth_str = p.depth.decode('utf-8', errors='ignore')
                    depth = float(depth_str) if depth_str.replace('.', '').replace('-', '').isdigit() else 0.0
                else:
                    depth = float(p.depth) if p.depth is not None else 0.0

                # Convert depth from meters to centimeters
                depth_cm = depth * 100

                if isinstance(p.area, bytes):
                    area_str = p.area.decode('utf-8', errors='ignore')
                    area = float(area_str) if area_str.replace('.', '').replace('-', '').isdigit() else 0.0
                else:
                    area = float(p.area) if p.area is not None else 0.0

                message += f"   📏 Depth: {depth_cm:.1f}cm | 📐 Area: {area:.0f}px\n"
            except (ValueError, AttributeError, UnicodeDecodeError) as e:
                logger.error(f"Error converting depth/area for pothole {i}: {e}")
                message += f"   📏 Depth: N/A | 📐 Area: N/A\n"

            message += f"   📍 Location: `{p.latitude:.4f}, {p.longitude:.4f}`\n"
            message += f"   🕒 {p.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"

        # Build pagination keyboard
        keyboard = []

        # Navigation buttons
        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton("⬅️ Previous", callback_data=f"severity:{severity}:{page - 1}"))
        if page < total_pages - 1:
            nav_buttons.append(InlineKeyboardButton("Next ➡️", callback_data=f"severity:{severity}:{page + 1}"))

        if nav_buttons:
            keyboard.append(nav_buttons)

        # Add location buttons for current page items
        for p in potholes[start_idx:end_idx]:
            location_text = f"📍 View on map: {p.city}"
            location_data = f"{p.latitude},{p.longitude}"
            keyboard.append([InlineKeyboardButton(location_text, callback_data=location_data)])

        # Add back button
        keyboard.append([InlineKeyboardButton("🔙 Back to Severity Menu", callback_data="back_to_severity")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def show_region_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show statistics for a specific region"""
        query = update.callback_query
        await query.answer()

        region = query.data.split(":")[1]

        # Get all potholes for the region
        potholes = self.db.get_potholes(filters={'region': region})

        if not potholes:
            await query.message.reply_text(f"No statistics available for {region}.")
            return

        # Calculate statistics
        severity_count = {}
        total_depth = 0
        total_area = 0

        for p in potholes:
            severity_count[p.severity.value] = severity_count.get(p.severity.value, 0) + 1

            # Handle depth conversion
            try:
                if isinstance(p.depth, bytes):
                    depth_str = p.depth.decode('utf-8', errors='ignore')
                    depth = float(depth_str) if depth_str.replace('.', '').replace('-', '').isdigit() else 0.0
                else:
                    depth = float(p.depth) if p.depth is not None else 0.0
                total_depth += depth
            except:
                pass

            # Handle area conversion
            try:
                if isinstance(p.area, bytes):
                    area_str = p.area.decode('utf-8', errors='ignore')
                    area = float(area_str) if area_str.replace('.', '').replace('-', '').isdigit() else 0.0
                else:
                    area = float(p.area) if p.area is not None else 0.0
                total_area += area
            except:
                pass

        avg_depth = total_depth / len(potholes)
        avg_depth_cm = avg_depth * 100  # Convert to cm
        avg_area = total_area / len(potholes)

        # Build message
        message = f"*📊 Statistics for {region}*\n\n"
        message += f"Total Potholes: {len(potholes)}\n\n"

        message += "*By Severity:*\n"
        severity_emojis = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in severity_count:
                emoji = severity_emojis.get(severity, '⚪')
                count = severity_count[severity]
                percentage = (count / len(potholes)) * 100
                message += f"{emoji} {severity.capitalize()}: {count} ({percentage:.1f}%)\n"

        message += f"\n*Average Measurements:*\n"
        message += f"📏 Average Depth: {avg_depth_cm:.1f}cm\n"
        message += f"📐 Average Area: {avg_area:.0f}px\n"



            # Back button
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data=f"region:{region}:all:0")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.message.edit_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def back_to_regions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Go back to region selection"""
        query = update.callback_query
        await query.answer()

        # Get all potholes to extract regions
        potholes = self.db.get_potholes()

        if not potholes:
            await query.message.edit_text("No pothole locations available.")
            return

        # Extract unique regions and count
        region_counts = {}
        for pothole in potholes:
            if pothole.region:
                region_counts[pothole.region] = region_counts.get(pothole.region, 0) + 1

        keyboard = []
        for region in sorted(region_counts.keys()):
            count = region_counts[region]
            button_text = f"{region} ({count} potholes)"
            callback_data = f"region:{region}:all:0"  # Default to show all
            keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            "Select a region to view pothole locations:",
            reply_markup=reply_markup
        )

    # Add a no-op handler for non-interactive buttons
    async def noop_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle non-interactive button presses"""
        query = update.callback_query
        await query.answer()

    async def send_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send specific location"""
        query = update.callback_query
        await query.answer()

        try:
            # Check if the data matches coordinate pattern
            if ',' not in query.data or query.data.startswith(('severity:', 'region:', 'back_')):
                logger.warning(f"Invalid location data received: {query.data}")
                return

            latitude, longitude = map(float, query.data.split(","))
            await query.message.reply_location(latitude=latitude, longitude=longitude)
        except ValueError as e:
            logger.error(f"Error parsing location: {e}, data: {query.data}")
            await query.message.reply_text("Invalid location data.")

    def run(self):
        """Start the bot"""
        self.application.run_polling()

    async def display_by_severity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Display potholes sorted by severity"""
        # Get statistics to show counts
        stats = self.db.get_statistics()
        severity_stats = stats.get('by_severity', {})

        keyboard = []

        # Add buttons with counts
        severity_emojis = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }

        for severity, emoji in severity_emojis.items():
            count = severity_stats.get(severity, 0)
            button_text = f"{emoji} {severity.capitalize()} ({count} potholes)"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"severity:{severity}")])

        # Add "All" button with total count
        total = stats.get('total', 0)
        keyboard.append([InlineKeyboardButton(f"📊 All Potholes ({total})", callback_data="severity:all")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "*🚨 View Potholes by Severity Level*\n\n"
            "Select a severity level to view detailed information:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def show_potholes_by_severity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show potholes filtered by severity with pagination"""
        query = update.callback_query
        await query.answer()

        # Parse callback data
        data_parts = query.data.split(":")
        severity = data_parts[1]
        page = int(data_parts[2]) if len(data_parts) > 2 else 0

        ITEMS_PER_PAGE = 5

        # Get potholes based on severity
        if severity == "all":
            potholes = self.db.get_potholes(sort_by='severity', sort_order='DESC')
            title = "All Potholes (sorted by severity)"
        else:
            potholes = self.db.get_potholes(filters={'severity': severity}, sort_by='depth', sort_order='DESC')
            title = f"{severity.capitalize()} Severity Potholes"

        if not potholes:
            await query.message.edit_text(f"No {severity} severity potholes found.")
            return

        # Calculate pagination
        total_pages = (len(potholes) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        start_idx = page * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, len(potholes))

        # Build message
        severity_emojis = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }

        message = f"*{title}*\n"
        message += f"_Page {page + 1} of {total_pages} • Total: {len(potholes)} potholes_\n\n"

        for i, p in enumerate(potholes[start_idx:end_idx], start=start_idx + 1):
            emoji = severity_emojis.get(p.severity.value, '⚪')
            message += f"{i}. {emoji} *{p.severity.value.upper()}* - {p.city}, {p.region}\n"

            # Handle depth and area with proper type conversion
            try:
                if isinstance(p.depth, bytes):
                    depth_str = p.depth.decode('utf-8', errors='ignore')
                    depth = float(depth_str) if depth_str.replace('.', '').replace('-', '').isdigit() else 0.0
                else:
                    depth = float(p.depth) if p.depth is not None else 0.0

                if isinstance(p.area, bytes):
                    area_str = p.area.decode('utf-8', errors='ignore')
                    area = float(area_str) if area_str.replace('.', '').replace('-', '').isdigit() else 0.0
                else:
                    area = float(p.area) if p.area is not None else 0.0

                message += f"   📏 Depth: {depth:.3f}m | 📐 Area: {area:.0f}px\n"
            except (ValueError, AttributeError, UnicodeDecodeError) as e:
                logger.error(f"Error converting depth/area for pothole {i}: {e}")
                message += f"   📏 Depth: N/A | 📐 Area: N/A\n"

            message += f"   📍 Location: `{p.latitude:.4f}, {p.longitude:.4f}`\n"
            message += f"   🕒 {p.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"

        # Build pagination keyboard (rest remains the same)
        keyboard = []

        # Navigation buttons
        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton("⬅️ Previous", callback_data=f"severity:{severity}:{page - 1}"))
        if page < total_pages - 1:
            nav_buttons.append(InlineKeyboardButton("Next ➡️", callback_data=f"severity:{severity}:{page + 1}"))

        if nav_buttons:
            keyboard.append(nav_buttons)

        # Add location buttons for current page items
        for p in potholes[start_idx:end_idx]:
            location_text = f"📍 View on map: {p.city}"
            location_data = f"{p.latitude},{p.longitude}"
            keyboard.append([InlineKeyboardButton(location_text, callback_data=location_data)])

        # Add back button
        keyboard.append([InlineKeyboardButton("🔙 Back to Severity Menu", callback_data="back_to_severity")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def back_to_severity_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Go back to severity selection menu"""
        query = update.callback_query
        await query.answer()

        # Reuse the display_by_severity logic
        stats = self.db.get_statistics()
        severity_stats = stats.get('by_severity', {})

        keyboard = []
        severity_emojis = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }

        for severity, emoji in severity_emojis.items():
            count = severity_stats.get(severity, 0)
            button_text = f"{emoji} {severity.capitalize()} ({count} potholes)"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"severity:{severity}")])

        total = stats.get('total', 0)
        keyboard.append([InlineKeyboardButton(f"📊 All Potholes ({total})", callback_data="severity:all")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            "*🚨 View Potholes by Severity Level*\n\n"
            "Select a severity level to view detailed information:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def export_csv(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        import os

        potholes = self.db.get_potholes()
        if not potholes:
            await update.message.reply_text("No data to export.")
            return

        # Convert to DataFrame
        data = [p.to_dict() for p in potholes]
        df = pd.DataFrame(data)

        # Create CSV file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"potholes_export_{timestamp}.csv"
        filepath = os.path.join(config.EXPORT_DIR, filename)

        df.to_csv(filepath, index=False)

        # Send file
        with open(filepath, 'rb') as f:
            await update.message.reply_document(
                document=f,
                filename=filename,
                caption=f"Pothole data export\nTotal records: {len(df)}"

            )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send detailed help message"""
        help_text = """
    *🚗 Pothole Detection Bot - Help Guide 🚗*

    This bot helps you track and monitor detected potholes in your area.

    *📋 Available Commands:*

    /start - Welcome message and quick overview
    /help - Show this detailed help guide
    /locations - Browse potholes by region with interactive menu
    /severity - View potholes filtered by severity level
    /map - Get a Google Maps link with all pothole locations
    /stats - View detection statistics and summary
    /export - Export all pothole data as CSV file

    *🎯 How to Use:*

    1️⃣ *Browse by Location:*
       • Use /locations to see regions with potholes
       • Select a region to view specific locations
       • Tap on a location to see it on the map

    2️⃣ *Filter by Severity:*
       • Use /severity to filter by severity level
       • 🟢 Low - Minor surface damage
       • 🟡 Medium - Moderate depth
       • 🟠 High - Significant hazard
       • 🔴 Critical - Severe road damage

    3️⃣ *View on Map:*
       • Use /map for a comprehensive map view
       • Opens in Google Maps with all locations
       • Plan routes avoiding problem areas

    4️⃣ *Export Data:*
       • Use /export to download CSV file
       • Contains all pothole information
       • Useful for reports and analysis

    *📊 Understanding the Data:*

    • *Depth*: Measured depth of pothole in meters
    • *Area*: Size of detected pothole in pixels
    • *Confidence*: Detection accuracy (0-100%)
    • *Timestamp*: When the pothole was detected

    *🔔 Tips:*
    • Potholes within 10m are considered duplicates
    • Data is updated in real-time from video feed
    • GPS coordinates are included when available

    *❓ Questions or Issues?*
    Contact the system administrator for support.

    Stay safe on the roads! 🛣️
    """

        await update.message.reply_text(help_text, parse_mode='Markdown')

    # Alternative: Shorter help command with inline keyboard for topics
    async def help_command_interactive(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send interactive help menu"""
        keyboard = [
            [InlineKeyboardButton("📍 Navigation Commands", callback_data="help:navigation")],
            [InlineKeyboardButton("📊 Data & Statistics", callback_data="help:data")],
            [InlineKeyboardButton("🎯 Using Filters", callback_data="help:filters")],
            [InlineKeyboardButton("💡 Tips & Tricks", callback_data="help:tips")],
            [InlineKeyboardButton("❓ FAQ", callback_data="help:faq")]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        help_text = """
    *🚗 Pothole Detection Bot - Help Center 🚗*

    Welcome to the help center! Select a topic below to learn more:

    • *Navigation* - Learn about location and map commands
    • *Data & Statistics* - Understanding pothole data
    • *Filters* - How to filter by severity and region
    • *Tips* - Best practices and useful features
    • *FAQ* - Frequently asked questions

    Or use /start to see all available commands.
    """

        await update.message.reply_text(
            help_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def help_topic_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle help topic selection"""
        query = update.callback_query
        await query.answer()

        topic = query.data.split(":")[1]

        help_topics = {
            "navigation": """
    *📍 Navigation Commands*

    • /locations - Browse potholes by region
      - Shows all regions with pothole counts
      - Select a region to see specific locations
      - Tap locations to view on map

    • /map - View all potholes on Google Maps
      - Opens comprehensive map view
      - Shows all detected locations
      - Useful for route planning

    • Individual Locations:
      - Each location shows coordinates
      - Tap to open in Telegram's map view
      - Share locations with others
    """,
            "data": """
    *📊 Data & Statistics*

    • /stats - View summary statistics
      - Total potholes detected
      - Breakdown by severity
      - Top affected regions

    • /export - Export data as CSV
      - Downloads complete dataset
      - Includes all pothole details
      - Perfect for reports/analysis

    • Data Fields:
      - Location (GPS coordinates)
      - Severity level
      - Depth in meters
      - Detection confidence
      - Timestamp
    """,
            "filters": """
    *🎯 Using Filters*

    • /severity - Filter by severity level
      - 🟢 Low: Minor damage (< 2cm)
      - 🟡 Medium: Moderate (2-5cm)
      - 🟠 High: Significant (5-10cm)
      - 🔴 Critical: Severe (> 10cm)

    • Regional Filtering:
      - Use /locations to filter by region
      - See counts per region
      - Focus on specific areas

    • Sorting:
      - Potholes sorted by severity
      - Most recent detections first
      - Deepest potholes prioritized
    """,
            "tips": """
    *💡 Tips & Tricks*

    • Duplicate Prevention:
      - Potholes within 10m are merged
      - Prevents duplicate reports
      - Ensures accurate counts

    • Real-time Updates:
      - Data updates continuously
      - New detections added instantly
      - Check /stats for latest counts

    • Best Practices:
      - Export data regularly
      - Monitor high-severity areas
      - Share locations with authorities
      - Plan routes using /map
    """,
            "faq": """
    *❓ Frequently Asked Questions*

    *Q: How accurate is the detection?*
    A: Detection confidence is shown for each pothole. Higher confidence = more accurate.

    *Q: Why are some potholes missing?*
    A: Detection depends on video quality and GPS signal. Some may be filtered as duplicates.

    *Q: Can I report potholes manually?*
    A: This bot only shows automatically detected potholes from the video feed.

    *Q: How often is data updated?*
    A: Data updates in real-time as the detection system processes video footage.

    *Q: What does "Unknown" location mean?*
    A: GPS data was unavailable when the pothole was detected.
    """
        }

        message = help_topics.get(topic, "Topic not found.")

        # Add back button
        keyboard = [[InlineKeyboardButton("🔙 Back to Help Menu", callback_data="help:menu")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.message.edit_text(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def help_menu_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Return to main help menu"""
        query = update.callback_query
        await query.answer()

        keyboard = [
            [InlineKeyboardButton("📍 Navigation Commands", callback_data="help:navigation")],
            [InlineKeyboardButton("📊 Data & Statistics", callback_data="help:data")],
            [InlineKeyboardButton("🎯 Using Filters", callback_data="help:filters")],
            [InlineKeyboardButton("💡 Tips & Tricks", callback_data="help:tips")],
            [InlineKeyboardButton("❓ FAQ", callback_data="help:faq")]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        help_text = """
    *🚗 Pothole Detection Bot - Help Center 🚗*

    Welcome to the help center! Select a topic below to learn more:

    • *Navigation* - Learn about location and map commands
    • *Data & Statistics* - Understanding pothole data
    • *Filters* - How to filter by severity and region
    • *Tips* - Best practices and useful features
    • *FAQ* - Frequently asked questions

    Or use /start to see all available commands.
    """
        
        await query.message.edit_text (
            help_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
