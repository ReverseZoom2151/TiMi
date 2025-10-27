"""Setup script for TiMi package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="timi",
    version="0.1.0",
    author="TiMi Development Team",
    description="Rationality-Driven Multi-Agent System for Quantitative Financial Trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ReverseZoom2151/TiMi",
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    python_requires=">=3.9",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "openai>=1.10.0",
        "anthropic>=0.18.0",
        "ccxt>=4.2.0",
        "python-binance>=1.0.19",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "ta>=0.11.0",
        "aiohttp>=3.9.0",
        "sqlalchemy>=2.0.0",
        "alembic>=1.13.0",
        "prometheus-client>=0.19.0",
        "structlog>=24.1.0",
        "requests>=2.31.0",
        "python-dateutil>=2.8.2",
        "pytz>=2023.3",
        "tenacity>=8.2.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="trading algorithmic-trading quantitative-finance llm ai-agents multi-agent-system",
    project_urls={
        "Bug Reports": "https://github.com/ReverseZoom2151/TiMi/issues",
        "Source": "https://github.com/ReverseZoom2151/TiMi",
    },
)
