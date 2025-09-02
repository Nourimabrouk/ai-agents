"""
Data Formatting Utilities
Consistent formatting functions for dashboard displays
"""

import streamlit as st
from typing import Union, Optional
import locale
from datetime import datetime, timedelta
import numpy as np

# Try to set locale for proper formatting
try:
    locale.setlocale(locale.LC_ALL, '')
except locale.Error:
    # Fallback if locale setting fails
    pass

def format_currency(amount: Union[float, int], currency: str = "USD", show_symbol: bool = True) -> str:
    """
    Format currency amounts with proper symbols and thousand separators
    
    Args:
        amount: Numeric amount to format
        currency: Currency code (USD, EUR, etc.)
        show_symbol: Whether to show currency symbol
        
    Returns:
        Formatted currency string
    """
    if amount is None:
        return "N/A"
    
    # Currency symbols
    symbols = {
        "USD": "$",
        "EUR": "€", 
        "GBP": "£",
        "JPY": "¥",
        "CAD": "C$",
        "AUD": "A$"
    }
    
    symbol = symbols.get(currency, "$") if show_symbol else ""
    
    try:
        # Format with thousand separators
        if abs(amount) >= 1000000:
            # Millions
            return f"{symbol}{amount/1000000:.1f}M"
        elif abs(amount) >= 1000:
            # Thousands
            return f"{symbol}{amount/1000:.1f}K"
        else:
            # Regular amount
            return f"{symbol}{amount:,.2f}"
    except (ValueError, TypeError):
        return f"{symbol}0.00"

def format_percentage(value: Union[float, int], decimal_places: int = 1) -> str:
    """
    Format percentage values
    
    Args:
        value: Numeric value to format as percentage
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
    
    try:
        return f"{value:.{decimal_places}f}%"
    except (ValueError, TypeError):
        return "0.0%"

def format_number(value: Union[float, int], compact: bool = True) -> str:
    """
    Format large numbers with appropriate suffixes
    
    Args:
        value: Numeric value to format
        compact: Whether to use compact notation (K, M, B)
        
    Returns:
        Formatted number string
    """
    if value is None:
        return "N/A"
    
    try:
        if not compact:
            return f"{value:,}"
        
        if abs(value) >= 1000000000:
            # Billions
            return f"{value/1000000000:.1f}B"
        elif abs(value) >= 1000000:
            # Millions
            return f"{value/1000000:.1f}M"
        elif abs(value) >= 1000:
            # Thousands
            return f"{value/1000:.1f}K"
        else:
            # Regular number
            return f"{value:,}"
    except (ValueError, TypeError):
        return "0"

def format_duration(seconds: Union[float, int]) -> str:
    """
    Format duration in seconds to human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds is None:
        return "N/A"
    
    try:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"
    except (ValueError, TypeError):
        return "0s"

def format_date(date: datetime, format_type: str = "short") -> str:
    """
    Format datetime objects for display
    
    Args:
        date: DateTime object to format
        format_type: Type of formatting (short, long, relative)
        
    Returns:
        Formatted date string
    """
    if date is None:
        return "N/A"
    
    try:
        if format_type == "short":
            return date.strftime("%m/%d/%Y")
        elif format_type == "long":
            return date.strftime("%B %d, %Y at %I:%M %p")
        elif format_type == "relative":
            now = datetime.now()
            diff = now - date
            
            if diff.days > 0:
                return f"{diff.days} days ago"
            elif diff.seconds > 3600:
                hours = diff.seconds // 3600
                return f"{hours} hours ago"
            elif diff.seconds > 60:
                minutes = diff.seconds // 60
                return f"{minutes} minutes ago"
            else:
                return "Just now"
        else:
            return date.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, AttributeError):
        return "Invalid Date"

def format_file_size(size_bytes: int) -> str:
    """
    Format file sizes in bytes to human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes is None or size_bytes < 0:
        return "N/A"
    
    try:
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = int(np.floor(np.log(size_bytes) / np.log(1024)))
        p = np.power(1024, i)
        s = round(size_bytes / p, 2)
        
        return f"{s} {size_names[i]}"
    except (ValueError, TypeError, OverflowError):
        return "0 B"

def format_metric_delta(current: float, previous: float, 
                       format_type: str = "percentage", 
                       reverse_colors: bool = False) -> dict:
    """
    Calculate and format metric deltas for Streamlit metrics
    
    Args:
        current: Current value
        previous: Previous value for comparison
        format_type: How to format the delta (percentage, absolute, currency)
        reverse_colors: Whether to reverse color logic (for metrics where lower is better)
        
    Returns:
        Dictionary with delta value and color
    """
    if current is None or previous is None or previous == 0:
        return {"value": "N/A", "color": "normal"}
    
    try:
        delta = current - previous
        
        if format_type == "percentage":
            delta_pct = (delta / previous) * 100
            delta_str = f"{delta_pct:+.1f}%"
        elif format_type == "currency":
            delta_str = format_currency(delta, show_symbol=True)
            if delta >= 0:
                delta_str = f"+{delta_str}"
        else:  # absolute
            delta_str = f"{delta:+,.0f}"
        
        # Determine color (green for positive, red for negative)
        if reverse_colors:
            color = "inverse" if delta > 0 else "normal"
        else:
            color = "normal"
        
        return {"value": delta_str, "color": color}
        
    except (ValueError, TypeError, ZeroDivisionError):
        return {"value": "N/A", "color": "normal"}

def format_status_indicator(status: str, show_icon: bool = True) -> str:
    """
    Format status indicators with colors and icons
    
    Args:
        status: Status string (healthy, warning, critical, etc.)
        show_icon: Whether to include status icons
        
    Returns:
        Formatted status string with emoji icons
    """
    status_lower = status.lower()
    
    status_map = {
        "healthy": {"icon": "🟢", "text": "Healthy"},
        "warning": {"icon": "🟡", "text": "Warning"},
        "critical": {"icon": "🔴", "text": "Critical"},
        "offline": {"icon": "⚫", "text": "Offline"},
        "online": {"icon": "🟢", "text": "Online"},
        "processing": {"icon": "🔵", "text": "Processing"},
        "completed": {"icon": "✅", "text": "Completed"},
        "failed": {"icon": "❌", "text": "Failed"},
        "pending": {"icon": "⏳", "text": "Pending"},
        "active": {"icon": "🟢", "text": "Active"},
        "inactive": {"icon": "🔴", "text": "Inactive"}
    }
    
    status_info = status_map.get(status_lower, {"icon": "❓", "text": status.title()})
    
    if show_icon:
        return f"{status_info['icon']} {status_info['text']}"
    else:
        return status_info['text']

def format_confidence_score(score: float, show_category: bool = True) -> str:
    """
    Format confidence scores with categories
    
    Args:
        score: Confidence score (0-100)
        show_category: Whether to show confidence category
        
    Returns:
        Formatted confidence score string
    """
    if score is None:
        return "N/A"
    
    try:
        score_str = f"{score:.1f}%"
        
        if show_category:
            if score >= 95:
                category = "🟢 Excellent"
            elif score >= 85:
                category = "🟡 Good"
            elif score >= 70:
                category = "🟠 Fair"
            else:
                category = "🔴 Poor"
            
            return f"{score_str} ({category})"
        else:
            return score_str
            
    except (ValueError, TypeError):
        return "N/A"

def format_throughput(value: Union[float, int], unit: str = "docs", 
                     time_period: str = "hour") -> str:
    """
    Format throughput metrics
    
    Args:
        value: Throughput value
        unit: Unit of measurement (docs, transactions, etc.)
        time_period: Time period (second, minute, hour, day)
        
    Returns:
        Formatted throughput string
    """
    if value is None:
        return "N/A"
    
    try:
        formatted_value = format_number(value, compact=True)
        return f"{formatted_value} {unit}/{time_period}"
    except (ValueError, TypeError):
        return f"0 {unit}/{time_period}"

def create_metric_card(title: str, value: str, delta: Optional[str] = None,
                      status: Optional[str] = None, help_text: Optional[str] = None) -> None:
    """
    Create a styled metric card with consistent formatting
    
    Args:
        title: Metric title
        value: Main metric value
        delta: Change indicator
        status: Status indicator
        help_text: Help text for tooltip
    """
    # Determine card style based on status
    if status:
        status_lower = status.lower()
        if status_lower in ["critical", "failed", "error"]:
            border_color = "#dc3545"
        elif status_lower in ["warning", "pending"]:
            border_color = "#ffc107"
        elif status_lower in ["healthy", "completed", "active"]:
            border_color = "#28a745"
        else:
            border_color = "#17a2b8"
    else:
        border_color = "#2a5298"
    
    # Create the metric card HTML
    card_html = f"""
    <div style="
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid {border_color};
        margin: 10px 0;
    ">
        <h4 style="margin: 0 0 10px 0; color: #333;">{title}</h4>
        <h2 style="margin: 0; color: {border_color};">{value}</h2>
        {f'<p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{delta}</p>' if delta else ''}
        {f'<p style="margin: 5px 0 0 0; color: #666; font-size: 12px;">{status}</p>' if status else ''}
    </div>
    """
    
    if help_text:
        with st.container():
            st.markdown(card_html, unsafe_allow_html=True)
            st.caption(help_text)
    else:
        st.markdown(card_html, unsafe_allow_html=True)

def format_data_table(df, format_config: dict = None) -> None:
    """
    Format and display data tables with consistent styling
    
    Args:
        df: Pandas DataFrame to display
        format_config: Dictionary of column formatting configurations
    """
    if df is None or df.empty:
        st.warning("No data available to display")
        return
    
    # Apply formatting if configuration provided
    if format_config:
        formatted_df = df.copy()
        
        for column, format_type in format_config.items():
            if column in formatted_df.columns:
                if format_type == "currency":
                    formatted_df[column] = formatted_df[column].apply(format_currency)
                elif format_type == "percentage":
                    formatted_df[column] = formatted_df[column].apply(format_percentage)
                elif format_type == "number":
                    formatted_df[column] = formatted_df[column].apply(format_number)
                elif format_type == "date":
                    formatted_df[column] = formatted_df[column].apply(
                        lambda x: format_date(x) if x else "N/A"
                    )
        
        st.dataframe(formatted_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

# Color scheme constants for consistent styling
DASHBOARD_COLORS = {
    'primary': '#2a5298',
    'secondary': '#1e3c72',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

def get_color_for_metric(metric_name: str, value: float = None) -> str:
    """
    Get appropriate color for a metric based on its name and value
    
    Args:
        metric_name: Name of the metric
        value: Optional value for threshold-based coloring
        
    Returns:
        Color code string
    """
    metric_lower = metric_name.lower()
    
    # Define color rules based on metric type
    if 'error' in metric_lower or 'fail' in metric_lower:
        return DASHBOARD_COLORS['danger']
    elif 'warning' in metric_lower or 'alert' in metric_lower:
        return DASHBOARD_COLORS['warning'] 
    elif 'success' in metric_lower or 'complete' in metric_lower:
        return DASHBOARD_COLORS['success']
    elif 'accuracy' in metric_lower and value:
        if value >= 95:
            return DASHBOARD_COLORS['success']
        elif value >= 85:
            return DASHBOARD_COLORS['warning']
        else:
            return DASHBOARD_COLORS['danger']
    else:
        return DASHBOARD_COLORS['primary']