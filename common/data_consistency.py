"""
Data Consistency Checking and Repair

Provides utilities for checking and repairing data consistency issues.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class ConsistencyIssue:
    """Represents a data consistency issue."""
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_data: str
    timestamp: datetime
    auto_repairable: bool = False


@dataclass
class RepairResult:
    """Result of a repair operation."""
    success: bool
    issue: ConsistencyIssue
    message: str
    timestamp: datetime


class DataConsistencyChecker:
    """
    Checks and repairs data consistency issues.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        self.consistency_checks: List[Callable] = []
        self.repair_actions: Dict[str, Callable] = {}
        self.issues: List[ConsistencyIssue] = []
        self.repair_history: List[RepairResult] = []
    
    def add_consistency_check(self, check_func: Callable) -> None:
        """Add a consistency check function."""
        self.consistency_checks.append(check_func)
        logger.info(f"Added consistency check: {check_func.__name__}")
    
    def add_repair_action(self, issue_type: str, repair_func: Callable) -> None:
        """Add a repair action for a specific issue type."""
        self.repair_actions[issue_type] = repair_func
        logger.info(f"Added repair action for: {issue_type}")

    
    async def run_consistency_checks(self) -> List[ConsistencyIssue]:
        """Run all consistency checks and return found issues."""
        self.issues = []
        
        for check_func in self.consistency_checks:
            try:
                if asyncio.iscoroutinefunction(check_func):
                    issues = await check_func()
                else:
                    issues = check_func()
                
                if issues:
                    if isinstance(issues, list):
                        self.issues.extend(issues)
                    else:
                        self.issues.append(issues)
                        
            except Exception as e:
                logger.error(f"Consistency check {check_func.__name__} failed: {e}")
                self.issues.append(ConsistencyIssue(
                    issue_type="check_failure",
                    severity="high",
                    description=f"Check {check_func.__name__} failed: {str(e)}",
                    affected_data=check_func.__name__,
                    timestamp=datetime.now()
                ))
        
        logger.info(f"Found {len(self.issues)} consistency issues")
        return self.issues
    
    async def repair_issue(self, issue: ConsistencyIssue) -> RepairResult:
        """Attempt to repair a consistency issue."""
        if issue.issue_type not in self.repair_actions:
            return RepairResult(
                success=False,
                issue=issue,
                message=f"No repair action available for {issue.issue_type}",
                timestamp=datetime.now()
            )
        
        repair_func = self.repair_actions[issue.issue_type]
        
        try:
            logger.info(f"Attempting to repair: {issue.description}")
            
            if asyncio.iscoroutinefunction(repair_func):
                success = await repair_func(issue)
            else:
                success = repair_func(issue)
            
            result = RepairResult(
                success=success,
                issue=issue,
                message="Repair successful" if success else "Repair failed",
                timestamp=datetime.now()
            )
            
            self.repair_history.append(result)
            
            if success:
                logger.info(f"Successfully repaired: {issue.description}")
            else:
                logger.warning(f"Failed to repair: {issue.description}")
            
            return result
            
        except Exception as e:
            logger.error(f"Repair action failed with exception: {e}")
            result = RepairResult(
                success=False,
                issue=issue,
                message=f"Repair exception: {str(e)}",
                timestamp=datetime.now()
            )
            self.repair_history.append(result)
            return result
    
    async def repair_all_issues(self, auto_only: bool = True) -> Dict[str, int]:
        """Repair all found issues."""
        if not self.issues:
            logger.info("No issues to repair")
            return {"total": 0, "repaired": 0, "failed": 0, "skipped": 0}
        
        stats = {"total": len(self.issues), "repaired": 0, "failed": 0, "skipped": 0}
        
        for issue in self.issues:
            if auto_only and not issue.auto_repairable:
                stats["skipped"] += 1
                continue
            
            result = await self.repair_issue(issue)
            if result.success:
                stats["repaired"] += 1
            else:
                stats["failed"] += 1
        
        logger.info(
            f"Repair summary: {stats['repaired']} repaired, "
            f"{stats['failed']} failed, {stats['skipped']} skipped"
        )
        
        return stats
    
    def get_consistency_report(self) -> Dict[str, Any]:
        """Get a report of consistency status."""
        issues_by_severity = {}
        for issue in self.issues:
            severity = issue.severity
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append({
                "type": issue.issue_type,
                "description": issue.description,
                "affected_data": issue.affected_data,
                "auto_repairable": issue.auto_repairable
            })
        
        return {
            "total_issues": len(self.issues),
            "issues_by_severity": issues_by_severity,
            "repair_history_count": len(self.repair_history),
            "last_check": datetime.now().isoformat()
        }


class FileIntegrityChecker:
    """
    Checks file integrity using checksums.
    """
    
    def __init__(self, manifest_file: Optional[Path] = None):
        self.manifest_file = manifest_file or Path("data/file_manifest.json")
        self.manifest: Dict[str, str] = {}
        self.load_manifest()
    
    def load_manifest(self) -> None:
        """Load file manifest from disk."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    self.manifest = json.load(f)
                logger.info(f"Loaded manifest with {len(self.manifest)} files")
            except Exception as e:
                logger.error(f"Failed to load manifest: {e}")
                self.manifest = {}
        else:
            logger.info("No existing manifest found")
    
    def save_manifest(self) -> None:
        """Save file manifest to disk."""
        try:
            self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_file, 'w') as f:
                json.dump(self.manifest, f, indent=2)
            logger.info(f"Saved manifest with {len(self.manifest)} files")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def add_file(self, file_path: Path) -> None:
        """Add a file to the manifest."""
        checksum = self.calculate_checksum(file_path)
        if checksum:
            self.manifest[str(file_path)] = checksum
            logger.debug(f"Added {file_path} to manifest")
    
    def verify_file(self, file_path: Path) -> bool:
        """Verify a file's integrity against the manifest."""
        file_str = str(file_path)
        
        if file_str not in self.manifest:
            logger.warning(f"File {file_path} not in manifest")
            return False
        
        expected_checksum = self.manifest[file_str]
        actual_checksum = self.calculate_checksum(file_path)
        
        if actual_checksum != expected_checksum:
            logger.error(
                f"Checksum mismatch for {file_path}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )
            return False
        
        return True
    
    def verify_all_files(self) -> List[str]:
        """Verify all files in the manifest."""
        corrupted_files = []
        
        for file_str in self.manifest.keys():
            file_path = Path(file_str)
            if not file_path.exists():
                logger.error(f"File missing: {file_path}")
                corrupted_files.append(file_str)
            elif not self.verify_file(file_path):
                corrupted_files.append(file_str)
        
        if corrupted_files:
            logger.warning(f"Found {len(corrupted_files)} corrupted/missing files")
        else:
            logger.info("All files verified successfully")
        
        return corrupted_files
