"""
Phase 1 Infrastructure Validation Tests

This test suite validates that all Phase 1 infrastructure is properly configured:
- Directory structure exists
- Dependencies are installable
- Configuration files are valid
- Docker services can be started
- Environment settings load correctly

These tests ensure we have a solid foundation before proceeding to Phase 2.

Author: Homunculus Team
"""

import os
import sys
from pathlib import Path
import pytest
import json
import yaml
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDirectoryStructure:
    """Test that all required directories exist."""
    
    def test_src_directories_exist(self):
        """Verify all source code directories are created."""
        base_path = Path(__file__).parent.parent.parent / "src" / "arena"
        
        expected_dirs = [
            "orchestration",
            "agents", 
            "message_bus",
            "scoring",
            "persistence",
            "models",
            "config",
            "config/prompts",
            "scenarios",
            "scenarios/templates",
            "cli",
            "utils"
        ]
        
        for dir_name in expected_dirs:
            dir_path = base_path / dir_name
            assert dir_path.exists(), f"Directory {dir_path} does not exist"
            assert dir_path.is_dir(), f"{dir_path} is not a directory"
            
            # Check for __init__.py
            init_file = dir_path / "__init__.py"
            assert init_file.exists(), f"Missing __init__.py in {dir_path}"
    
    def test_test_directories_exist(self):
        """Verify test directories are created."""
        base_path = Path(__file__).parent
        
        expected_dirs = [
            "test_agents",
            "test_scoring",
            "test_message_bus",
            "test_models",
            "test_orchestration"
        ]
        
        for dir_name in expected_dirs:
            dir_path = base_path / dir_name
            assert dir_path.exists(), f"Test directory {dir_path} does not exist"
            assert dir_path.is_dir(), f"{dir_path} is not a directory"
    
    def test_config_directories_exist(self):
        """Verify configuration directories are created."""
        base_path = Path(__file__).parent.parent.parent
        
        paths_to_check = [
            base_path / "configs" / "arena" / "scenarios",
            base_path / "scripts" / "arena",
            base_path / "data" / "arena" / "logs"
        ]
        
        for path in paths_to_check:
            assert path.exists(), f"Directory {path} does not exist"
            assert path.is_dir(), f"{path} is not a directory"


class TestDependencies:
    """Test that all required dependencies are specified."""
    
    def test_pyproject_toml_exists(self):
        """Verify pyproject.toml exists and is valid."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"
        
        # Try to parse it
        try:
            import toml
            with open(pyproject_path) as f:
                data = toml.load(f)
                assert "tool" in data
                assert "poetry" in data["tool"]
        except ImportError:
            # If toml not installed, at least check file is readable
            with open(pyproject_path) as f:
                content = f.read()
                assert "kafka-python" in content, "kafka-python dependency not found"
                assert "langgraph" in content, "langgraph dependency not found"
                assert "psycopg2-binary" in content, "psycopg2-binary dependency not found"
    
    def test_arena_dependencies_listed(self):
        """Verify Arena-specific dependencies are listed."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        
        with open(pyproject_path) as f:
            content = f.read()
            
        required_deps = [
            "kafka-python",
            "langgraph", 
            "psycopg2-binary",
            "pydantic",
            "langchain-core",
            "redis",
            "chromadb"
        ]
        
        for dep in required_deps:
            assert dep in content, f"Required dependency {dep} not found in pyproject.toml"


class TestDockerConfiguration:
    """Test Docker configuration files."""
    
    def test_docker_compose_arena_exists(self):
        """Verify docker-compose.arena.yml exists and is valid."""
        docker_path = Path(__file__).parent.parent.parent / "docker-compose.arena.yml"
        assert docker_path.exists(), "docker-compose.arena.yml not found"
        
        # Try to parse YAML
        with open(docker_path) as f:
            try:
                data = yaml.safe_load(f)
                assert "services" in data
                assert "kafka" in data["services"]
                assert "zookeeper" in data["services"]
                assert "postgres" in data["services"]
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in docker-compose.arena.yml: {e}")
    
    def test_postgres_init_script_exists(self):
        """Verify PostgreSQL initialization script exists."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "arena" / "init_db.sql"
        assert script_path.exists(), "PostgreSQL init script not found"
        
        # Check it contains expected tables
        with open(script_path) as f:
            content = f.read()
            
        expected_tables = [
            "CREATE TABLE IF NOT EXISTS games",
            "CREATE TABLE IF NOT EXISTS participants",
            "CREATE TABLE IF NOT EXISTS messages",
            "CREATE TABLE IF NOT EXISTS scoring_history",
            "CREATE TABLE IF NOT EXISTS accusations",
            "CREATE TABLE IF NOT EXISTS eliminations",
            "CREATE TABLE IF NOT EXISTS champion_history"
        ]
        
        for table in expected_tables:
            assert table in content, f"Table definition '{table}' not found in init script"
    
    def test_docker_services_configuration(self):
        """Verify Docker services are properly configured."""
        docker_path = Path(__file__).parent.parent.parent / "docker-compose.arena.yml"
        
        with open(docker_path) as f:
            data = yaml.safe_load(f)
        
        # Check Kafka configuration
        kafka = data["services"]["kafka"]
        assert "depends_on" in kafka
        assert "zookeeper" in kafka["depends_on"]
        assert "9092:9092" in kafka["ports"]
        
        # Check PostgreSQL configuration  
        postgres = data["services"]["postgres"]
        assert "POSTGRES_DB" in postgres["environment"]
        assert "5432:5432" in postgres["ports"] or "${POSTGRES_PORT:-5432}:5432" in postgres["ports"]
        
        # Check health checks exist
        for service in ["kafka", "zookeeper", "postgres"]:
            assert "healthcheck" in data["services"][service], f"No healthcheck for {service}"


class TestEnvironmentConfiguration:
    """Test environment configuration files."""
    
    def test_env_arena_example_exists(self):
        """Verify .env.arena.example exists and contains required variables."""
        env_path = Path(__file__).parent.parent.parent / ".env.arena.example"
        assert env_path.exists(), ".env.arena.example not found"
        
        with open(env_path) as f:
            content = f.read()
        
        # Check for required environment variables
        required_vars = [
            "KAFKA_BOOTSTRAP_SERVERS",
            "KAFKA_TOPIC_PREFIX",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "REDIS_HOST",
            "REDIS_PORT",
            "REDIS_DB",
            "DEFAULT_LLM_PROVIDER",
            "MIN_AGENTS",
            "MAX_AGENTS",
            "DEFAULT_ELIMINATION_THRESHOLD",
            "DEFAULT_GAME_THEORY_MODE"
        ]
        
        for var in required_vars:
            assert var in content, f"Required environment variable {var} not in .env.arena.example"
    
    def test_settings_module_imports(self):
        """Verify arena_settings.py can be imported."""
        try:
            from src.arena.config.arena_settings import ArenaSettings, settings
            
            # Check settings object is created
            assert settings is not None
            assert isinstance(settings, ArenaSettings)
            
            # Check default values
            assert settings.kafka_bootstrap_servers == "localhost:9092"
            assert settings.kafka_topic_prefix == "arena"
            assert settings.min_agents == 2
            assert settings.max_agents == 8
            
        except ImportError as e:
            pytest.fail(f"Failed to import arena_settings: {e}")
    
    def test_settings_validation(self):
        """Test that settings validation works correctly."""
        from src.arena.config.arena_settings import ArenaSettings
        
        # Test invalid scoring weights (don't sum to 1.0)
        with pytest.raises(ValueError, match="Scoring weights must sum to 1.0"):
            ArenaSettings(
                score_weight_novelty=0.5,
                score_weight_builds=0.5,
                score_weight_solves=0.5,
                score_weight_radical=0.5,
                score_weight_manipulation=0.5
            )
        
        # Test valid configuration
        settings = ArenaSettings(
            score_weight_novelty=0.2,
            score_weight_builds=0.2,
            score_weight_solves=0.2,
            score_weight_radical=0.2,
            score_weight_manipulation=0.2
        )
        assert abs(sum(settings.scoring_weights.values()) - 1.0) < 0.01
    
    def test_settings_url_properties(self):
        """Test URL property generation."""
        from src.arena.config.arena_settings import ArenaSettings
        
        settings = ArenaSettings(
            postgres_host="dbhost",
            postgres_port=5433,
            postgres_user="user",
            postgres_password="pass",
            postgres_db="mydb"
        )
        
        expected_url = "postgresql://user:pass@dbhost:5433/mydb"
        assert settings.postgres_url == expected_url
        
        # Test Redis URL
        settings = ArenaSettings(
            redis_host="redishost",
            redis_port=6380,
            redis_db=2
        )
        assert "redis://redishost:6380/2" in settings.redis_url


class TestModuleImports:
    """Test that all Arena modules can be imported."""
    
    def test_arena_main_module_imports(self):
        """Test importing the main arena module."""
        try:
            import src.arena
            assert src.arena.__version__ == "1.0.0"
            assert src.arena.__author__ == "Homunculus Team"
        except ImportError as e:
            pytest.fail(f"Failed to import src.arena: {e}")
    
    def test_submodule_imports(self):
        """Test that all submodules can be imported."""
        modules = [
            "src.arena.orchestration",
            "src.arena.agents",
            "src.arena.message_bus",
            "src.arena.scoring",
            "src.arena.persistence",
            "src.arena.models",
            "src.arena.config",
            "src.arena.scenarios",
            "src.arena.cli",
            "src.arena.utils"
        ]
        
        for module_name in modules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")
    
    def test_module_docstrings(self):
        """Test that all modules have proper docstrings."""
        modules = [
            "src.arena",
            "src.arena.orchestration",
            "src.arena.agents",
            "src.arena.message_bus",
            "src.arena.scoring",
            "src.arena.persistence",
            "src.arena.models"
        ]
        
        for module_name in modules:
            module = __import__(module_name, fromlist=[''])
            assert module.__doc__ is not None, f"Module {module_name} missing docstring"
            assert len(module.__doc__.strip()) > 20, f"Module {module_name} has insufficient docstring"


class TestFilePermissions:
    """Test file permissions and accessibility."""
    
    def test_log_directory_writable(self):
        """Test that log directory is writable."""
        log_dir = Path(__file__).parent.parent.parent / "data" / "arena" / "logs"
        
        # Try to create a test file
        test_file = log_dir / "test_write.tmp"
        try:
            test_file.write_text("test")
            assert test_file.exists()
            test_file.unlink()  # Clean up
        except Exception as e:
            pytest.fail(f"Cannot write to log directory: {e}")
    
    def test_scripts_executable(self):
        """Test that shell scripts have proper permissions."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts" / "arena"
        
        # Check if any .sh files exist and are executable
        sh_files = list(scripts_dir.glob("*.sh"))
        for script in sh_files:
            # On Unix systems, check execute permission
            if os.name != 'nt':  # Not Windows
                assert os.access(script, os.X_OK), f"Script {script} is not executable"


class TestIntegrationReadiness:
    """Test that the system is ready for Phase 2 integration."""
    
    def test_can_connect_to_services(self):
        """Test connectivity to required services (when available)."""
        import socket
        
        # This test is informational - services might not be running yet
        services = [
            ("localhost", 9092, "Kafka"),
            ("localhost", 5432, "PostgreSQL"),
            ("localhost", 6379, "Redis"),
            ("localhost", 2181, "Zookeeper")
        ]
        
        results = []
        for host, port, name in services:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                results.append(f"✓ {name} is accessible on {host}:{port}")
            else:
                results.append(f"✗ {name} is not accessible on {host}:{port} (expected if Docker not running)")
        
        # Print results for information
        print("\nService Connectivity Check:")
        for result in results:
            print(f"  {result}")
    
    def test_phase1_checklist_complete(self):
        """Verify all Phase 1 tasks are complete."""
        checklist = {
            "Directory structure created": self._check_directories(),
            "Dependencies updated": self._check_dependencies(), 
            "Docker configuration ready": self._check_docker_config(),
            "Environment configuration ready": self._check_env_config(),
            "Modules importable": self._check_imports(),
            "Settings validation working": self._check_settings()
        }
        
        print("\nPhase 1 Completion Checklist:")
        all_complete = True
        for task, complete in checklist.items():
            status = "✓" if complete else "✗"
            print(f"  {status} {task}")
            if not complete:
                all_complete = False
        
        assert all_complete, "Not all Phase 1 tasks are complete"
    
    def _check_directories(self) -> bool:
        """Check if all directories exist."""
        base_path = Path(__file__).parent.parent.parent / "src" / "arena"
        return all([
            (base_path / "orchestration").exists(),
            (base_path / "agents").exists(),
            (base_path / "models").exists()
        ])
    
    def _check_dependencies(self) -> bool:
        """Check if dependencies are specified."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if not pyproject_path.exists():
            return False
        with open(pyproject_path) as f:
            content = f.read()
        return "kafka-python" in content and "langgraph" in content
    
    def _check_docker_config(self) -> bool:
        """Check if Docker config exists."""
        docker_path = Path(__file__).parent.parent.parent / "docker-compose.arena.yml"
        return docker_path.exists()
    
    def _check_env_config(self) -> bool:
        """Check if environment config exists."""
        env_path = Path(__file__).parent.parent.parent / ".env.arena.example"
        return env_path.exists()
    
    def _check_imports(self) -> bool:
        """Check if modules can be imported."""
        try:
            import src.arena
            from src.arena.config.arena_settings import settings
            return True
        except ImportError:
            return False
    
    def _check_settings(self) -> bool:
        """Check if settings work."""
        try:
            from src.arena.config.arena_settings import ArenaSettings
            settings = ArenaSettings()
            # Settings work if they load successfully
            # API keys are optional for development
            return True
        except Exception:
            return False


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])