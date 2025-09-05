"""
Authentication and Authorization Management
Handles user authentication, session management, and role-based access
"""

import streamlit as st
import hashlib
import json
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
import secrets

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.secret_key = os.getenv("DASHBOARD_SECRET_KEY", "dashboard-secret-key-change-in-production")
        self.session_timeout_hours = 24
        
        # Default users (in production, this would be from a database)
        self.users = {
            "admin@company.com": {
                "password": self._hash_password("admin123"),
                "role": "admin",
                "name": "System Administrator",
                "organization": "All Organizations",
                "permissions": ["read", "write", "delete", "admin"]
            },
            "manager@company.com": {
                "password": self._hash_password("manager123"),
                "role": "manager", 
                "name": "Dashboard Manager",
                "organization": "Acme Corp",
                "permissions": ["read", "write"]
            },
            "user@company.com": {
                "password": self._hash_password("user123"),
                "role": "user",
                "name": "Dashboard User",
                "organization": "Acme Corp", 
                "permissions": ["read"]
            },
            "demo@company.com": {
                "password": self._hash_password("demo123"),
                "role": "demo",
                "name": "Demo User",
                "organization": "Demo Corp",
                "permissions": ["read"]
            }
        }
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize authentication session state"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_info' not in st.session_state:
            st.session_state.user_info = None
        if 'session_token' not in st.session_state:
            st.session_state.session_token = None
        if 'session_expires' not in st.session_state:
            st.session_state.session_expires = None
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_token(self, user_email: str) -> Tuple[str, datetime]:
        """Generate JWT token for authenticated user"""
        expires = datetime.utcnow() + timedelta(hours=self.session_timeout_hours)
        
        payload = {
            'user_email': user_email,
            'exp': expires,
            'iat': datetime.utcnow(),
            'jti': secrets.token_hex(16)  # Unique token ID
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        return token, expires
    
    def _verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return {}
        except jwt.InvalidTokenError:
            return {}
    
    def authenticate(self, email: str, password: str) -> bool:
        """Authenticate user with email and password"""
        if email not in self.users:
            return False
        
        stored_password = self.users[email]['password']
        provided_password = self._hash_password(password)
        
        if stored_password == provided_password:
            # Generate session token
            token, expires = self._generate_token(email)
            
            # Update session state
            st.session_state.authenticated = True
            st.session_state.user_info = {
                'email': email,
                'name': self.users[email]['name'],
                'role': self.users[email]['role'],
                'organization': self.users[email]['organization'],
                'permissions': self.users[email]['permissions']
            }
            st.session_state.session_token = token
            st.session_state.session_expires = expires
            
            return True
        
        return False
    
    def logout(self):
        """Logout current user"""
        st.session_state.authenticated = False
        st.session_state.user_info = None
        st.session_state.session_token = None
        st.session_state.session_expires = None
    
    def is_authenticated(self) -> bool:
        """Check if current user is authenticated"""
        if not st.session_state.authenticated:
            return False
        
        if not st.session_state.session_token:
            return False
        
        # Verify token is still valid
        payload = self._verify_token(st.session_state.session_token)
        if not payload:
            self.logout()
            return False
        
        return True
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current authenticated user info"""
        if self.is_authenticated():
            return st.session_state.user_info
        return {}
    
    def has_permission(self, permission: str) -> bool:
        """Check if current user has specific permission"""
        user_info = self.get_current_user()
        if not user_info:
            return False
        
        return permission in user_info.get('permissions', [])
    
    def require_permission(self, permission: str) -> bool:
        """Require specific permission, show error if not authorized"""
        if not self.has_permission(permission):
            st.error(f"❌ Access Denied: You need '{permission}' permission to access this feature.")
            return False
        return True
    
    def render_login_form(self):
        """Render login form"""
        st.markdown("""
        <div style="
            max-width: 400px;
            margin: 50px auto;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background: white;
        ">
        <h2 style="text-align: center; color: #2a5298; margin-bottom: 30px;">
            🚀 Enterprise Dashboard Login
        </h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("### 🔐 Please Login to Continue")
            
            email = st.text_input("📧 Email Address", placeholder="Enter your email")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("🚀 Login", type="primary", use_container_width=True)
            with col2:
                demo_button = st.form_submit_button("👀 Demo Access", use_container_width=True)
            
            if login_button:
                if email and password:
                    if self.authenticate(email, password):
                        st.success("✅ Login successful! Welcome back!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password. Please try again.")
                else:
                    st.warning("⚠️ Please enter both email and password.")
            
            elif demo_button:
                if self.authenticate("demo@company.com", "demo123"):
                    st.success("✅ Demo access granted! Exploring dashboard...")
                    st.rerun()
        
        # Display demo credentials
        with st.expander("🔓 Demo Credentials", expanded=False):
            st.markdown("""
            **Demo Account:**
            - Email: demo@company.com
            - Password: demo123
            - Role: Read-only access
            
            **Manager Account:**
            - Email: manager@company.com  
            - Password: manager123
            - Role: Read/Write access
            
            **Admin Account:**
            - Email: admin@company.com
            - Password: admin123
            - Role: Full access
            """)
    
    def render_user_profile(self):
        """Render user profile in sidebar"""
        user_info = self.get_current_user()
        if not user_info:
            return {}
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 User Profile")
        
        # User info
        st.sidebar.markdown(f"**Name:** {user_info['name']}")
        st.sidebar.markdown(f"**Email:** {user_info['email']}")
        st.sidebar.markdown(f"**Role:** {user_info['role'].title()}")
        st.sidebar.markdown(f"**Organization:** {user_info['organization']}")
        
        # Session info
        if st.session_state.session_expires:
            time_remaining = st.session_state.session_expires - datetime.utcnow()
            hours_remaining = int(time_remaining.total_seconds() / 3600)
            st.sidebar.markdown(f"**Session:** {hours_remaining}h remaining")
        
        # Permissions
        permissions = user_info.get('permissions', [])
        permission_icons = {
            'read': '👁️ View',
            'write': '✏️ Edit', 
            'delete': '🗑️ Delete',
            'admin': '⚙️ Admin'
        }
        
        st.sidebar.markdown("**Permissions:**")
        for perm in permissions:
            icon_text = permission_icons.get(perm, f'🔹 {perm}')
            st.sidebar.markdown(f"• {icon_text}")
        
        # Logout button
        if st.sidebar.button("🚪 Logout", type="secondary", use_container_width=True):
            self.logout()
            st.rerun()
    
    def check_role_access(self, required_role: str) -> bool:
        """Check if user has required role or higher"""
        user_info = self.get_current_user()
        if not user_info:
            return False
        
        role_hierarchy = ['user', 'demo', 'manager', 'admin']
        user_role = user_info.get('role', 'user')
        
        try:
            user_level = role_hierarchy.index(user_role)
            required_level = role_hierarchy.index(required_role)
            return user_level >= required_level
        except ValueError:
            return False
    
    def get_organization_filter(self) -> Optional[str]:
        """Get organization filter for current user"""
        user_info = self.get_current_user()
        if not user_info:
            return {}
        
        organization = user_info.get('organization')
        if organization == "All Organizations":
            return None  # No filter for admin users
        
        return organization
    
    def render_role_based_content(self, content_config: Dict[str, Any]):
        """Render content based on user role"""
        user_info = self.get_current_user()
        if not user_info:
            return {}
        
        user_role = user_info.get('role', 'user')
        
        # Show content based on role
        if user_role in content_config:
            content = content_config[user_role]
            if callable(content):
                content()
            else:
                st.markdown(content)
    
    def audit_log(self, action: str, details: str = ""):
        """Log user actions for audit trail"""
        user_info = self.get_current_user()
        if not user_info:
            return {}
        
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user': user_info['email'],
            'role': user_info['role'],
            'organization': user_info['organization'],
            'action': action,
            'details': details,
            'ip_address': 'localhost',  # In production, get real IP
            'user_agent': 'Streamlit Dashboard'  # In production, get real user agent
        }
        
        # In production, this would be logged to a secure audit database
        # For now, we'll just store in session state
        if 'audit_log' not in st.session_state:
            st.session_state.audit_log = []
        
        st.session_state.audit_log.append(log_entry)
        
        # Keep only last 100 entries to prevent memory issues
        if len(st.session_state.audit_log) > 100:
            st.session_state.audit_log = st.session_state.audit_log[-100:]

# Global auth manager instance
_auth_manager = None

def get_auth_manager() -> AuthManager:
    """Get global auth manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager