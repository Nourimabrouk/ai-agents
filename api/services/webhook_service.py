"""
Webhook Service
Comprehensive webhook management and delivery system
"""

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp
from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession

from api.database.session import get_database_session
from api.models.database_models import Webhook, WebhookDelivery, Document
from api.models.api_models import WebhookConfig, WebhookEvent, WebhookPayload
from api.config import get_settings
from utils.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class WebhookDeliveryError(Exception):
    """Custom exception for webhook delivery failures"""
    pass


class WebhookService:
    """
    Comprehensive webhook service providing:
    - Event-driven webhook notifications
    - Reliable delivery with retry logic
    - HMAC signature verification
    - Delivery status tracking
    - Webhook health monitoring
    - Rate limiting and throttling
    - Batch delivery optimization
    """
    
    def __init__(self):
        self.settings = settings
        
        # Delivery state
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.delivery_workers: List[asyncio.Task] = []
        self.is_running = False
        
        # HTTP session for webhook deliveries
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking
        self.delivery_stats = {
            "total_deliveries": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "total_retry_attempts": 0
        }
        
        # Rate limiting per webhook URL
        self.rate_limits: Dict[str, List[float]] = {}
        
        # Failed webhook URLs (temporary blacklist)
        self.failed_webhooks: Dict[str, datetime] = {}
        
        logger.info("WebhookService initialized")
    
    async def initialize(self):
        """Initialize webhook service"""
        try:
            # Create HTTP session
            connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
            timeout = aiohttp.ClientTimeout(total=settings.integrations.webhook_timeout)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            
            # Start delivery workers
            await self.start_delivery_workers()
            
            logger.info("Webhook service initialized successfully")
            
        except Exception as e:
            logger.error(f"Webhook service initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup webhook service resources"""
        try:
            # Stop delivery workers
            await self.stop_delivery_workers()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            logger.info("Webhook service cleanup completed")
            
        except Exception as e:
            logger.error(f"Webhook service cleanup failed: {e}")
    
    async def start_delivery_workers(self, worker_count: int = 3):
        """Start webhook delivery worker tasks"""
        if self.is_running:
            logger.warning("Webhook delivery workers already running")
            return {}
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(worker_count):
            worker = asyncio.create_task(self._delivery_worker(f"worker_{i}"))
            self.delivery_workers.append(worker)
        
        logger.info(f"Started {worker_count} webhook delivery workers")
    
    async def stop_delivery_workers(self):
        """Stop webhook delivery workers"""
        if not self.is_running:
            return {}
        
        self.is_running = False
        
        # Cancel all workers
        for worker in self.delivery_workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.delivery_workers:
            await asyncio.gather(*self.delivery_workers, return_exceptions=True)
        
        self.delivery_workers.clear()
        logger.info("Webhook delivery workers stopped")
    
    async def create_webhook(
        self,
        webhook_config: WebhookConfig,
        user_id: str,
        organization_id: str
    ) -> Dict[str, Any]:
        """Create new webhook configuration"""
        try:
            async with get_database_session() as db:
                # Validate URL
                parsed_url = urlparse(str(webhook_config.url))
                if not parsed_url.scheme or not parsed_url.netloc:
                    raise ValueError("Invalid webhook URL")
                
                # Create webhook record
                webhook = Webhook(
                    organization_id=organization_id,
                    created_by_id=user_id,
                    name=f"webhook_{int(time.time())}",
                    url=str(webhook_config.url),
                    secret=webhook_config.secret,
                    events=[event.value for event in webhook_config.events],
                    headers=webhook_config.headers or {},
                    timeout_seconds=webhook_config.timeout_seconds,
                    retry_attempts=webhook_config.retry_attempts,
                    is_active=webhook_config.active
                )
                
                db.add(webhook)
                await db.commit()
                await db.refresh(webhook)
                
                # Test webhook delivery
                if webhook_config.active:
                    await self._test_webhook(webhook)
                
                return {
                    "id": str(webhook.id),
                    "url": webhook.url,
                    "events": webhook.events,
                    "active": webhook.is_active,
                    "created_at": webhook.created_at.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Webhook creation failed: {e}")
            raise
    
    async def list_webhooks(self, organization_id: str) -> List[Dict[str, Any]]:
        """List all webhooks for organization"""
        try:
            async with get_database_session() as db:
                result = await db.execute(
                    select(Webhook).where(Webhook.organization_id == organization_id)
                )
                webhooks = result.scalars().all()
                
                return [
                    {
                        "id": str(webhook.id),
                        "name": webhook.name,
                        "url": webhook.url,
                        "events": webhook.events,
                        "is_active": webhook.is_active,
                        "total_deliveries": webhook.total_deliveries,
                        "successful_deliveries": webhook.successful_deliveries,
                        "failed_deliveries": webhook.failed_deliveries,
                        "last_delivery_at": webhook.last_delivery_at.isoformat() if webhook.last_delivery_at else None,
                        "created_at": webhook.created_at.isoformat()
                    }
                    for webhook in webhooks
                ]
                
        except Exception as e:
            logger.error(f"Failed to list webhooks: {e}")
            return []
    
    async def delete_webhook(self, webhook_id: str, organization_id: str) -> bool:
        """Delete webhook"""
        try:
            async with get_database_session() as db:
                result = await db.execute(
                    select(Webhook).where(
                        and_(Webhook.id == webhook_id, Webhook.organization_id == organization_id)
                    )
                )
                webhook = result.scalar_one_or_none()
                
                if not webhook:
                    return False
                
                await db.delete(webhook)
                await db.commit()
                
                logger.info(f"Webhook deleted: {webhook_id}")
                return True
                
        except Exception as e:
            logger.error(f"Webhook deletion failed: {e}")
            return False
    
    async def send_webhook(
        self,
        event_type: WebhookEvent,
        data: Dict[str, Any],
        organization_id: str,
        user_id: Optional[str] = None
    ):
        """Send webhook notification for event"""
        try:
            # Get active webhooks for this organization and event
            async with get_database_session() as db:
                result = await db.execute(
                    select(Webhook).where(
                        and_(
                            Webhook.organization_id == organization_id,
                            Webhook.is_active == True,
                            Webhook.events.contains([event_type.value])
                        )
                    )
                )
                webhooks = result.scalars().all()
                
                if not webhooks:
                    logger.debug(f"No active webhooks found for event {event_type.value}")
                    return {}
                
                # Create webhook payload
                payload = WebhookPayload(
                    event_type=event_type,
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    data=data,
                    organization_id=organization_id,
                    user_id=user_id
                )
                
                # Queue delivery for each webhook
                for webhook in webhooks:
                    await self._queue_webhook_delivery(webhook, payload)
                
        except Exception as e:
            logger.error(f"Webhook sending failed: {e}")
    
    # Convenience methods for common events
    
    async def send_processing_complete_webhook(
        self,
        webhook_url: str,
        processing_result: Dict[str, Any]
    ):
        """Send processing complete webhook to specific URL"""
        try:
            # Create temporary webhook payload
            payload = WebhookPayload(
                event_type=WebhookEvent.DOCUMENT_PROCESSED,
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                data=processing_result,
                organization_id="",  # Will be filled from processing_result
                user_id=None
            )
            
            # Create delivery directly
            delivery = WebhookDelivery(
                webhook_id=None,  # Direct delivery
                event_type=WebhookEvent.DOCUMENT_PROCESSED.value,
                event_id=payload.event_id,
                payload=payload.dict(),
                scheduled_at=datetime.utcnow()
            )
            
            # Queue for delivery
            await self.delivery_queue.put({
                "delivery": delivery,
                "webhook_url": webhook_url,
                "secret": None,
                "headers": {},
                "timeout": settings.integrations.webhook_timeout,
                "max_attempts": 3
            })
            
        except Exception as e:
            logger.error(f"Direct webhook sending failed: {e}")
    
    async def send_document_processed_webhook(
        self,
        document_id: str,
        processing_result: Dict[str, Any],
        organization_id: str,
        user_id: str
    ):
        """Send document processed event"""
        await self.send_webhook(
            WebhookEvent.DOCUMENT_PROCESSED,
            {
                "document_id": document_id,
                "processing_result": processing_result
            },
            organization_id,
            user_id
        )
    
    async def send_document_failed_webhook(
        self,
        document_id: str,
        error_message: str,
        organization_id: str,
        user_id: str
    ):
        """Send document processing failed event"""
        await self.send_webhook(
            WebhookEvent.DOCUMENT_FAILED,
            {
                "document_id": document_id,
                "error_message": error_message
            },
            organization_id,
            user_id
        )
    
    async def send_batch_completed_webhook(
        self,
        batch_id: str,
        batch_results: Dict[str, Any],
        organization_id: str,
        user_id: str
    ):
        """Send batch processing completed event"""
        await self.send_webhook(
            WebhookEvent.BATCH_COMPLETED,
            {
                "batch_id": batch_id,
                "batch_results": batch_results
            },
            organization_id,
            user_id
        )
    
    # Private methods
    
    async def _delivery_worker(self, worker_name: str):
        """Worker task for delivering webhooks"""
        logger.info(f"Webhook delivery worker {worker_name} started")
        
        try:
            while self.is_running:
                try:
                    # Get delivery from queue
                    delivery_data = await asyncio.wait_for(
                        self.delivery_queue.get(),
                        timeout=1.0
                    )
                    
                    # Deliver webhook
                    await self._deliver_webhook(delivery_data)
                    
                except asyncio.TimeoutError:
                    continue  # Check if still running
                except Exception as e:
                    logger.error(f"Delivery worker {worker_name} error: {e}")
                    
        except asyncio.CancelledError:
            logger.info(f"Webhook delivery worker {worker_name} cancelled")
        except Exception as e:
            logger.error(f"Webhook delivery worker {worker_name} failed: {e}")
    
    async def _queue_webhook_delivery(self, webhook: Webhook, payload: WebhookPayload):
        """Queue webhook delivery"""
        try:
            async with get_database_session() as db:
                # Create delivery record
                delivery = WebhookDelivery(
                    webhook_id=webhook.id,
                    event_type=payload.event_type.value,
                    event_id=payload.event_id,
                    payload=payload.dict(),
                    max_attempts=webhook.retry_attempts,
                    scheduled_at=datetime.utcnow()
                )
                
                db.add(delivery)
                await db.commit()
                await db.refresh(delivery)
                
                # Add to delivery queue
                await self.delivery_queue.put({
                    "delivery": delivery,
                    "webhook_url": webhook.url,
                    "secret": webhook.secret,
                    "headers": webhook.headers,
                    "timeout": webhook.timeout_seconds,
                    "max_attempts": webhook.retry_attempts
                })
                
        except Exception as e:
            logger.error(f"Failed to queue webhook delivery: {e}")
    
    async def _deliver_webhook(self, delivery_data: Dict[str, Any]):
        """Deliver single webhook with retry logic"""
        delivery = delivery_data["delivery"]
        webhook_url = delivery_data["webhook_url"]
        
        try:
            # Check if URL is temporarily blacklisted
            if self._is_webhook_blacklisted(webhook_url):
                logger.debug(f"Skipping blacklisted webhook: {webhook_url}")
                return {}
            
            # Check rate limiting
            if self._is_rate_limited(webhook_url):
                logger.debug(f"Rate limited webhook: {webhook_url}")
                # Re-queue for later
                await asyncio.sleep(60)  # Wait 1 minute
                await self.delivery_queue.put(delivery_data)
                return {}
            
            # Prepare request
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"Enterprise-Document-Processing-API/1.0",
                **delivery_data.get("headers", {})
            }
            
            # Add HMAC signature if secret provided
            payload_json = json.dumps(delivery.payload)
            if delivery_data.get("secret"):
                signature = self._generate_hmac_signature(
                    payload_json,
                    delivery_data["secret"]
                )
                headers["X-Webhook-Signature"] = signature
            
            # Make HTTP request
            start_time = time.time()
            
            async with self.session.post(
                webhook_url,
                data=payload_json,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=delivery_data["timeout"])
            ) as response:
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Update delivery record
                await self._update_delivery_record(
                    delivery,
                    response.status,
                    await response.text(),
                    dict(response.headers),
                    duration_ms,
                    None
                )
                
                # Update statistics
                self.delivery_stats["total_deliveries"] += 1
                
                if 200 <= response.status < 300:
                    # Success
                    self.delivery_stats["successful_deliveries"] += 1
                    await self._update_webhook_stats(delivery.webhook_id, True)
                    
                    logger.info(
                        f"Webhook delivered successfully: {webhook_url} (status: {response.status})",
                        extra={
                            "webhook_url": webhook_url,
                            "event_type": delivery.event_type,
                            "event_id": delivery.event_id,
                            "status_code": response.status,
                            "duration_ms": duration_ms
                        }
                    )
                else:
                    # HTTP error
                    await self._handle_delivery_failure(delivery_data, f"HTTP {response.status}")
                
        except aiohttp.ClientTimeout:
            await self._handle_delivery_failure(delivery_data, "Request timeout")
            
        except aiohttp.ClientError as e:
            await self._handle_delivery_failure(delivery_data, f"Client error: {str(e)}")
            
        except Exception as e:
            await self._handle_delivery_failure(delivery_data, f"Unexpected error: {str(e)}")
    
    async def _handle_delivery_failure(self, delivery_data: Dict[str, Any], error_message: str):
        """Handle webhook delivery failure with retry logic"""
        delivery = delivery_data["delivery"]
        webhook_url = delivery_data["webhook_url"]
        
        try:
            # Update delivery record with error
            await self._update_delivery_record(
                delivery,
                None,
                None,
                None,
                None,
                error_message
            )
            
            # Update statistics
            self.delivery_stats["failed_deliveries"] += 1
            
            # Check if we should retry
            delivery.attempt_count += 1
            
            if delivery.attempt_count < delivery_data["max_attempts"]:
                # Schedule retry with exponential backoff
                retry_delay = min(300, 2 ** delivery.attempt_count * 10)  # Max 5 minutes
                
                self.delivery_stats["total_retry_attempts"] += 1
                
                logger.warning(
                    f"Webhook delivery failed, retrying in {retry_delay}s: {webhook_url}",
                    extra={
                        "webhook_url": webhook_url,
                        "event_id": delivery.event_id,
                        "attempt": delivery.attempt_count,
                        "max_attempts": delivery.max_attempts,
                        "error": error_message
                    }
                )
                
                # Re-queue with delay
                asyncio.create_task(self._delayed_retry(delivery_data, retry_delay))
                
            else:
                # Max attempts reached, mark as failed
                logger.error(
                    f"Webhook delivery permanently failed after {delivery.attempt_count} attempts: {webhook_url}",
                    extra={
                        "webhook_url": webhook_url,
                        "event_id": delivery.event_id,
                        "error": error_message
                    }
                )
                
                # Temporarily blacklist problematic webhooks
                self._blacklist_webhook(webhook_url)
                
                await self._update_webhook_stats(delivery.webhook_id, False)
            
        except Exception as e:
            logger.error(f"Failed to handle delivery failure: {e}")
    
    async def _delayed_retry(self, delivery_data: Dict[str, Any], delay_seconds: int):
        """Schedule delayed retry"""
        await asyncio.sleep(delay_seconds)
        await self.delivery_queue.put(delivery_data)
    
    async def _update_delivery_record(
        self,
        delivery: WebhookDelivery,
        status_code: Optional[int],
        response_body: Optional[str],
        response_headers: Optional[Dict[str, str]],
        duration_ms: Optional[int],
        error_message: Optional[str]
    ):
        """Update webhook delivery record in database"""
        try:
            async with get_database_session() as db:
                delivery.response_status_code = status_code
                delivery.response_body = response_body[:1000] if response_body else None  # Truncate
                delivery.response_headers = response_headers
                delivery.duration_ms = duration_ms
                delivery.error_message = error_message
                
                if status_code and 200 <= status_code < 300:
                    delivery.status = "delivered"
                    delivery.delivered_at = datetime.utcnow()
                elif error_message:
                    delivery.status = "failed"
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Failed to update delivery record: {e}")
    
    async def _update_webhook_stats(self, webhook_id: Optional[str], success: bool):
        """Update webhook statistics"""
        if not webhook_id:
            return {}
        
        try:
            async with get_database_session() as db:
                webhook = await db.get(Webhook, webhook_id)
                if webhook:
                    webhook.total_deliveries += 1
                    if success:
                        webhook.successful_deliveries += 1
                        webhook.last_success_at = datetime.utcnow()
                    else:
                        webhook.failed_deliveries += 1
                    
                    webhook.last_delivery_at = datetime.utcnow()
                    await db.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update webhook stats: {e}")
    
    async def _test_webhook(self, webhook: Webhook):
        """Test webhook with ping event"""
        try:
            test_payload = WebhookPayload(
                event_type=WebhookEvent.DOCUMENT_PROCESSED,  # Use as test event
                event_id=f"test_{int(time.time())}",
                timestamp=datetime.utcnow(),
                data={"test": True, "message": "Webhook test"},
                organization_id=str(webhook.organization_id),
                user_id=None
            )
            
            await self._queue_webhook_delivery(webhook, test_payload)
            
        except Exception as e:
            logger.error(f"Webhook test failed: {e}")
    
    def _generate_hmac_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload"""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    def _is_rate_limited(self, webhook_url: str) -> bool:
        """Check if webhook URL is rate limited"""
        current_time = time.time()
        
        # Clean old entries
        if webhook_url in self.rate_limits:
            self.rate_limits[webhook_url] = [
                req_time for req_time in self.rate_limits[webhook_url]
                if current_time - req_time < 60  # 1 minute window
            ]
        else:
            self.rate_limits[webhook_url] = []
        
        # Check rate limit (max 60 requests per minute per URL)
        if len(self.rate_limits[webhook_url]) >= 60:
            return True
        
        # Add current request
        self.rate_limits[webhook_url].append(current_time)
        return False
    
    def _is_webhook_blacklisted(self, webhook_url: str) -> bool:
        """Check if webhook URL is temporarily blacklisted"""
        if webhook_url not in self.failed_webhooks:
            return False
        
        # Check if blacklist period has expired (1 hour)
        blacklist_time = self.failed_webhooks[webhook_url]
        if datetime.utcnow() - blacklist_time > timedelta(hours=1):
            del self.failed_webhooks[webhook_url]
            return False
        
        return True
    
    def _blacklist_webhook(self, webhook_url: str):
        """Temporarily blacklist webhook URL"""
        self.failed_webhooks[webhook_url] = datetime.utcnow()
        logger.warning(f"Webhook temporarily blacklisted: {webhook_url}")
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get webhook delivery statistics"""
        success_rate = 0.0
        if self.delivery_stats["total_deliveries"] > 0:
            success_rate = (
                self.delivery_stats["successful_deliveries"] / 
                self.delivery_stats["total_deliveries"]
            ) * 100
        
        return {
            **self.delivery_stats,
            "success_rate_percent": round(success_rate, 2),
            "queue_size": self.delivery_queue.qsize(),
            "active_workers": len(self.delivery_workers),
            "blacklisted_urls": len(self.failed_webhooks)
        }


# Export service class
__all__ = ["WebhookService", "WebhookDeliveryError"]