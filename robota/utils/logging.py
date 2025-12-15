import os
import sys
from pathlib import Path
from loguru import logger

__all__ = [
    'Logger',
    'default_logger',
]

ROBOTA_DIR = os.getenv("ROBOTA_DIR") if os.getenv("ROBOTA_DIR") else os.path.expanduser('~/.robota')


class Logger:
    """日志管理类，支持控制台和文件输出"""
    
    def __init__(
        self,
        log_dir: str = os.path.join(ROBOTA_DIR, "logs"),
        log_file: str = "robota.log",
        rotation: str = "10 MB",
        retention: str = "7 days",
        level: str = "INFO",
        console_output: bool = True
    ):
        """
        初始化日志器
        
        Args:
            log_dir: 日志目录路径
            log_file: 日志文件名
            rotation: 日志轮转大小，如 "10 MB", "1 GB", "500 KB"
            retention: 日志保留时间，如 "7 days", "1 week", "1 month"
            level: 日志级别，如 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            console_output: 是否输出到控制台
        """
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.rotation = rotation
        self.retention = retention
        self.level = level
        self.console_output = console_output
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 移除默认的日志处理器
        logger.remove()
        
        # 配置日志格式
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # 添加控制台输出
        if self.console_output:
            logger.add(
                sys.stderr,
                format=log_format,
                level=self.level,
                colorize=True
            )
        
        # 添加文件输出
        log_path = self.log_dir / self.log_file
        logger.add(
            str(log_path),
            format=log_format,
            level=self.level,
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",  # 压缩旧日志
            encoding="utf-8",
            enqueue=True  # 异步写入，线程安全
        )
        
        self.logger = logger
    
    def get_logger(self):
        """获取日志器实例"""
        return self.logger
    
    def debug(self, message: str, **kwargs):
        """记录 DEBUG 级别日志"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """记录 INFO 级别日志"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录 WARNING 级别日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录 ERROR 级别日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """记录 CRITICAL 级别日志"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """记录异常信息"""
        self.logger.exception(message, **kwargs)


# 创建默认日志器实例
default_logger = Logger()