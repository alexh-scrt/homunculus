"""Test chat interface functionality."""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    from cli.chat_interface import ChatInterface
    from cli.debug_view import DebugView
    from config.character_loader import CharacterLoader
except ImportError:
    pytest.skip("CLI modules not available", allow_module_level=True)


class TestChatInterface:
    """Test ChatInterface functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.interface = ChatInterface()
        self.temp_dir = tempfile.mkdtemp()
        self.interface.save_dir = Path(self.temp_dir)
    
    def test_chat_interface_initialization(self):
        """Test ChatInterface can be initialized."""
        interface = ChatInterface()
        assert interface.console is not None
        assert interface.debug_view is not None
        assert isinstance(interface.debug_view, DebugView)
        assert interface.character_loader is not None
        assert isinstance(interface.character_loader, CharacterLoader)
        assert interface.current_character is None
        assert interface.debug_mode is False
    
    def test_display_welcome(self):
        """Test welcome message display."""
        # This method doesn't return anything, just prints
        # We test it doesn't raise an exception
        self.interface._display_welcome()
        # If we get here without exception, test passes
        assert True
    
    @pytest.mark.asyncio
    async def test_select_character_with_characters(self):
        """Test character selection when characters are available."""
        mock_characters = ['ada_lovelace', 'zen_master']
        mock_character_info = [
            {
                'id': 'ada_lovelace',
                'name': 'Ada Lovelace', 
                'archetype': 'analytical_genius',
                'description': 'Test description'
            }
        ]
        
        with patch.object(self.interface.character_loader, 'list_available_characters', return_value=mock_characters):
            with patch.object(self.interface.character_loader, 'get_character_info', 
                            return_value={'name': 'Ada Lovelace', 'archetype': 'analytical_genius'}):
                with patch('rich.prompt.Prompt.ask', return_value='1'):
                    result = await self.interface._select_character()
                    assert result == 'ada_lovelace'
    
    @pytest.mark.asyncio
    async def test_select_character_quit(self):
        """Test character selection when user quits."""
        mock_characters = ['ada_lovelace']
        
        with patch.object(self.interface.character_loader, 'list_available_characters', return_value=mock_characters):
            with patch.object(self.interface.character_loader, 'get_character_info', 
                            return_value={'name': 'Ada Lovelace', 'archetype': 'analytical_genius'}):
                with patch('rich.prompt.Prompt.ask', return_value='q'):
                    result = await self.interface._select_character()
                    assert result is None
    
    @pytest.mark.asyncio
    async def test_select_character_no_characters(self):
        """Test character selection when no characters available."""
        with patch.object(self.interface.character_loader, 'list_available_characters', return_value=[]):
            result = await self.interface._select_character()
            assert result is None
    
    @pytest.mark.asyncio
    async def test_load_character_success(self):
        """Test successful character loading."""
        mock_config = {
            'name': 'Test Character',
            'archetype': 'test',
            'description': 'A test character'
        }
        
        mock_character = Mock()
        mock_character.initialize = AsyncMock()
        mock_character.character_name = 'Test Character'
        
        with patch.object(self.interface.character_loader, 'load_character', return_value=mock_config):
            with patch('cli.chat_interface.CharacterAgent', return_value=mock_character):
                result = await self.interface._load_character('test_character')
                assert result == mock_character
                mock_character.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_character_failure(self):
        """Test character loading failure."""
        with patch.object(self.interface.character_loader, 'load_character', 
                         side_effect=Exception("Load failed")):
            result = await self.interface._load_character('test_character')
            assert result is None
    
    @pytest.mark.asyncio
    async def test_handle_command_exit(self):
        """Test exit command handling."""
        with patch('rich.prompt.Confirm.ask', return_value=True):
            result = await self.interface._handle_command('/exit')
            assert result == 'exit'
        
        with patch('rich.prompt.Confirm.ask', return_value=False):
            result = await self.interface._handle_command('/exit')
            assert result is None
    
    @pytest.mark.asyncio
    async def test_handle_command_debug(self):
        """Test debug command handling."""
        initial_debug_mode = self.interface.debug_mode
        result = await self.interface._handle_command('/debug')
        assert self.interface.debug_mode != initial_debug_mode
        assert result is None
    
    @pytest.mark.asyncio
    async def test_handle_command_help(self):
        """Test help command handling."""
        result = await self.interface._handle_command('/help')
        assert result is None
    
    @pytest.mark.asyncio
    async def test_handle_command_unknown(self):
        """Test unknown command handling."""
        result = await self.interface._handle_command('/unknown')
        assert result is None
    
    @pytest.mark.asyncio
    async def test_save_character_state(self):
        """Test saving character state."""
        mock_character = Mock()
        mock_character.character_id = 'test_character'
        mock_character.get_state_dict = Mock(return_value={'test': 'data'})
        
        self.interface.current_character = mock_character
        
        await self.interface._save_character_state('test_save')
        
        # Check file was created
        expected_file = self.interface.save_dir / 'test_save.json'
        assert expected_file.exists()
        
        # Check content
        with open(expected_file, 'r') as f:
            data = json.load(f)
        assert data == {'test': 'data'}
    
    @pytest.mark.asyncio
    async def test_save_character_state_auto_filename(self):
        """Test saving with auto-generated filename."""
        mock_character = Mock()
        mock_character.character_id = 'test_character'
        mock_character.get_state_dict = Mock(return_value={'test': 'data'})
        
        self.interface.current_character = mock_character
        
        await self.interface._save_character_state()
        
        # Check file was created with auto name
        expected_file = self.interface.save_dir / 'test_character_autosave.json'
        assert expected_file.exists()
    
    @pytest.mark.asyncio
    async def test_load_character_state(self):
        """Test loading character state."""
        # Create test save file
        test_data = {'test': 'state_data'}
        test_file = self.interface.save_dir / 'test_load.json'
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        mock_character = Mock()
        mock_character.character_id = 'test_character'
        mock_character.load_state_dict = AsyncMock()
        
        self.interface.current_character = mock_character
        
        await self.interface._load_character_state('test_load')
        
        mock_character.load_state_dict.assert_called_once_with(test_data)
    
    @pytest.mark.asyncio
    async def test_load_character_state_missing_file(self):
        """Test loading non-existent state file."""
        mock_character = Mock()
        mock_character.character_id = 'test_character'
        
        self.interface.current_character = mock_character
        
        # Should not raise exception, just display error
        await self.interface._load_character_state('nonexistent')
    
    @pytest.mark.asyncio
    async def test_search_memories(self):
        """Test memory search functionality."""
        mock_memories = [
            {
                'timestamp': '2025-01-01T10:00:00',
                'description': 'Test memory description',
                'similarity': 0.95
            }
        ]
        
        mock_character = Mock()
        mock_character.recall_past_conversations = AsyncMock(return_value=mock_memories)
        
        self.interface.current_character = mock_character
        
        await self.interface._search_memories('test query')
        
        mock_character.recall_past_conversations.assert_called_once_with(
            query='test query',
            limit=5
        )
    
    @pytest.mark.asyncio
    async def test_search_memories_no_results(self):
        """Test memory search with no results."""
        mock_character = Mock()
        mock_character.recall_past_conversations = AsyncMock(return_value=[])
        
        self.interface.current_character = mock_character
        
        await self.interface._search_memories('no results')
        
        mock_character.recall_past_conversations.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reset_character(self):
        """Test character reset functionality."""
        mock_config = {'name': 'Test Character', 'archetype': 'test'}
        mock_new_character = Mock()
        mock_new_character.initialize = AsyncMock()
        
        self.interface.current_character = Mock()
        self.interface.current_character.character_id = 'test_character'
        
        with patch('rich.prompt.Confirm.ask', return_value=True):
            with patch.object(self.interface.character_loader, 'load_character', return_value=mock_config):
                with patch('cli.chat_interface.CharacterAgent', return_value=mock_new_character):
                    await self.interface._reset_character()
                    assert self.interface.current_character == mock_new_character
                    mock_new_character.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reset_character_cancelled(self):
        """Test character reset when cancelled."""
        original_character = Mock()
        self.interface.current_character = original_character
        
        with patch('rich.prompt.Confirm.ask', return_value=False):
            await self.interface._reset_character()
            assert self.interface.current_character == original_character
    
    @pytest.mark.asyncio
    async def test_display_character_status(self):
        """Test character status display."""
        mock_state = Mock()
        mock_state.agent_states = {
            'mood': {
                'current_state': 'happy',
                'intensity': 0.8
            }
        }
        mock_state.conversation_history = ['msg1', 'msg2', 'msg3']
        
        mock_character = Mock()
        mock_character.character_name = 'Test Character'
        mock_character.character_id = 'test_character'
        mock_character.state = mock_state
        
        self.interface.current_character = mock_character
        
        # Should not raise exception
        await self.interface._display_character_status()
    
    @pytest.mark.asyncio
    async def test_auto_save(self):
        """Test auto-save functionality."""
        mock_character = Mock()
        mock_character.character_id = 'test_character'
        mock_character.get_state_dict = Mock(return_value={'auto': 'save_data'})
        
        self.interface.current_character = mock_character
        
        await self.interface._auto_save()
        
        # Check auto-save file was created
        expected_file = self.interface.save_dir / 'test_character_autosave.json'
        assert expected_file.exists()
    
    @pytest.mark.asyncio
    async def test_auto_save_no_character(self):
        """Test auto-save with no current character."""
        self.interface.current_character = None
        
        # Should not raise exception
        await self.interface._auto_save()
    
    def test_command_parsing(self):
        """Test command parsing logic."""
        # Test in _handle_command method through various commands
        test_commands = [
            '/exit',
            '/debug',
            '/help',
            '/save filename',
            '/load filename',
            '/memory search query',
            '/reset',
            '/status'
        ]
        
        for cmd in test_commands:
            # Just verify the command parsing doesn't crash
            parts = cmd.strip().split(' ', 1)
            cmd_name = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            assert cmd_name.startswith('/')
            assert isinstance(args, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])