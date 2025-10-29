"""
Data Export Module for Arena

This module provides functionality to export game data, analytics,
and reports in various formats for external analysis.

Features:
- Multiple export formats (JSON, CSV, Excel, PDF)
- Customizable export configurations
- Batch export capabilities
- Report generation
- Data visualization exports

Author: Homunculus Team
"""

import logging
import json
import csv
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import io

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Available export formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class ExportConfig:
    """Configuration for data export."""
    format: ExportFormat
    include_metadata: bool = True
    include_raw_data: bool = False
    include_analytics: bool = True
    include_visualizations: bool = False
    compression: bool = False
    
    # Filtering options
    date_range: Optional[Tuple[datetime, datetime]] = None
    agent_filter: Optional[List[str]] = None
    game_filter: Optional[List[str]] = None
    
    # Format-specific options
    csv_delimiter: str = ","
    json_indent: int = 2
    excel_sheet_names: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["format"] = self.format.value
        return data


class DataExporter:
    """
    Main data exporter for Arena.
    """
    
    def __init__(self, export_dir: str = "arena_exports"):
        """
        Initialize data exporter.
        
        Args:
            export_dir: Directory for exports
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        
    def export_game(
        self,
        game_data: Dict[str, Any],
        config: ExportConfig,
        filename: Optional[str] = None
    ) -> Path:
        """
        Export game data.
        
        Args:
            game_data: Game data to export
            config: Export configuration
            filename: Optional filename
            
        Returns:
            Path to exported file
        """
        if not filename:
            game_id = game_data.get("game_id", "unknown")
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"game_{game_id}_{timestamp}"
        
        # Filter data if needed
        filtered_data = self._filter_data(game_data, config)
        
        # Export based on format
        if config.format == ExportFormat.JSON:
            return self._export_json(filtered_data, filename, config)
        elif config.format == ExportFormat.CSV:
            return self._export_csv(filtered_data, filename, config)
        elif config.format == ExportFormat.EXCEL:
            return self._export_excel(filtered_data, filename, config)
        elif config.format == ExportFormat.HTML:
            return self._export_html(filtered_data, filename, config)
        elif config.format == ExportFormat.MARKDOWN:
            return self._export_markdown(filtered_data, filename, config)
        else:
            raise ValueError(f"Unsupported format: {config.format}")
    
    def _filter_data(
        self,
        data: Dict[str, Any],
        config: ExportConfig
    ) -> Dict[str, Any]:
        """Filter data based on configuration."""
        filtered = data.copy()
        
        # Apply date range filter
        if config.date_range:
            start, end = config.date_range
            # Filter based on timestamps in data
            # Implementation depends on data structure
        
        # Apply agent filter
        if config.agent_filter:
            if "agents" in filtered:
                filtered["agents"] = [
                    a for a in filtered["agents"]
                    if a.get("agent_id") in config.agent_filter
                ]
        
        # Remove raw data if not needed
        if not config.include_raw_data:
            filtered.pop("raw_messages", None)
            filtered.pop("raw_events", None)
        
        # Remove metadata if not needed
        if not config.include_metadata:
            filtered.pop("metadata", None)
            filtered.pop("config", None)
        
        return filtered
    
    def _export_json(
        self,
        data: Dict[str, Any],
        filename: str,
        config: ExportConfig
    ) -> Path:
        """Export as JSON."""
        file_path = self.export_dir / f"{filename}.json"
        
        # Convert non-serializable objects
        def default_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, set):
                return list(obj)
            elif hasattr(obj, "to_dict"):
                return obj.to_dict()
            else:
                return str(obj)
        
        with open(file_path, 'w') as f:
            json.dump(
                data,
                f,
                indent=config.json_indent,
                default=default_handler
            )
        
        if config.compression:
            import gzip
            gz_path = file_path.with_suffix(".json.gz")
            with open(file_path, 'rb') as f_in:
                with gzip.open(gz_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            file_path.unlink()  # Remove uncompressed file
            return gz_path
        
        return file_path
    
    def _export_csv(
        self,
        data: Dict[str, Any],
        filename: str,
        config: ExportConfig
    ) -> Path:
        """Export as CSV."""
        file_path = self.export_dir / f"{filename}.csv"
        
        # Flatten nested data for CSV
        rows = self._flatten_for_csv(data)
        
        if not rows:
            logger.warning("No data to export to CSV")
            return None
        
        # Write CSV
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys(),
                delimiter=config.csv_delimiter
            )
            writer.writeheader()
            writer.writerows(rows)
        
        return file_path
    
    def _flatten_for_csv(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten nested data structure for CSV export."""
        rows = []
        
        # Handle different data structures
        if "turns" in data:
            # Game data with turns
            for turn in data.get("turns", []):
                row = {
                    "game_id": data.get("game_id"),
                    "turn_number": turn.get("turn_number"),
                    "phase": turn.get("phase"),
                    "active_agents": turn.get("active_agents"),
                    "speaker": turn.get("speaker_id"),
                    "eliminated": ",".join(turn.get("eliminated", []))
                }
                rows.append(row)
        
        elif "agents" in data:
            # Agent data
            for agent in data.get("agents", []):
                row = {
                    "agent_id": agent.get("agent_id"),
                    "agent_name": agent.get("agent_name"),
                    "final_score": agent.get("final_score"),
                    "final_position": agent.get("final_position"),
                    "eliminated_turn": agent.get("elimination_turn"),
                    "is_winner": agent.get("is_champion", False)
                }
                rows.append(row)
        
        elif "scores" in data:
            # Score data
            for score_entry in data.get("scores", []):
                row = {
                    "agent_id": score_entry.get("agent_id"),
                    "turn": score_entry.get("turn_number"),
                    "score": score_entry.get("score"),
                    "score_change": score_entry.get("score_change")
                }
                rows.append(row)
        
        else:
            # Generic flattening
            rows = [self._flatten_dict(data)]
        
        return rows
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '_'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _export_excel(
        self,
        data: Dict[str, Any],
        filename: str,
        config: ExportConfig
    ) -> Path:
        """Export as Excel."""
        try:
            import pandas as pd
            import openpyxl
        except ImportError:
            logger.error("pandas and openpyxl required for Excel export")
            return self._export_csv(data, filename, config)
        
        file_path = self.export_dir / f"{filename}.xlsx"
        
        # Create Excel writer
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Export different data types to different sheets
            
            # Game summary sheet
            if "game_id" in data:
                summary_df = pd.DataFrame([{
                    "Game ID": data.get("game_id"),
                    "Start Time": data.get("start_time"),
                    "End Time": data.get("end_time"),
                    "Total Turns": data.get("total_turns"),
                    "Winner": data.get("winner_name"),
                    "Total Agents": data.get("total_agents")
                }])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Agents sheet
            if "agents" in data:
                agents_df = pd.DataFrame(data["agents"])
                agents_df.to_excel(writer, sheet_name="Agents", index=False)
            
            # Turns sheet
            if "turns" in data:
                turns_df = pd.DataFrame(data["turns"])
                turns_df.to_excel(writer, sheet_name="Turns", index=False)
            
            # Scores sheet
            if "scores" in data:
                scores_df = pd.DataFrame(data["scores"])
                scores_df.to_excel(writer, sheet_name="Scores", index=False)
            
            # Messages sheet (if included)
            if config.include_raw_data and "messages" in data:
                messages_df = pd.DataFrame(data["messages"])
                messages_df.to_excel(writer, sheet_name="Messages", index=False)
        
        return file_path
    
    def _export_html(
        self,
        data: Dict[str, Any],
        filename: str,
        config: ExportConfig
    ) -> Path:
        """Export as HTML."""
        file_path = self.export_dir / f"{filename}.html"
        
        html_content = self._generate_html_report(data, config)
        
        with open(file_path, 'w') as f:
            f.write(html_content)
        
        return file_path
    
    def _generate_html_report(
        self,
        data: Dict[str, Any],
        config: ExportConfig
    ) -> str:
        """Generate HTML report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Arena Game Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; border-bottom: 1px solid #ccc; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .summary { background: #f9f9f9; padding: 15px; border-radius: 5px; }
                .winner { color: gold; font-weight: bold; }
            </style>
        </head>
        <body>
        """
        
        # Game summary
        html += f"""
        <h1>Arena Game Report</h1>
        <div class="summary">
            <p><strong>Game ID:</strong> {data.get('game_id', 'N/A')}</p>
            <p><strong>Date:</strong> {data.get('start_time', 'N/A')}</p>
            <p><strong>Total Turns:</strong> {data.get('total_turns', 0)}</p>
            <p><strong>Winner:</strong> <span class="winner">{data.get('winner_name', 'N/A')}</span></p>
        </div>
        """
        
        # Agents table
        if "agents" in data:
            html += """
            <h2>Agents</h2>
            <table>
                <tr>
                    <th>Agent</th>
                    <th>Final Score</th>
                    <th>Position</th>
                    <th>Status</th>
                </tr>
            """
            
            for agent in data["agents"]:
                status = "Winner" if agent.get("is_champion") else "Eliminated"
                html += f"""
                <tr>
                    <td>{agent.get('agent_name', agent.get('agent_id'))}</td>
                    <td>{agent.get('final_score', 0):.2f}</td>
                    <td>{agent.get('final_position', 'N/A')}</td>
                    <td>{status}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Analytics section
        if config.include_analytics and "analytics" in data:
            html += """
            <h2>Analytics</h2>
            <div class="summary">
            """
            
            analytics = data["analytics"]
            for key, value in analytics.items():
                html += f"<p><strong>{key}:</strong> {value}</p>"
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _export_markdown(
        self,
        data: Dict[str, Any],
        filename: str,
        config: ExportConfig
    ) -> Path:
        """Export as Markdown."""
        file_path = self.export_dir / f"{filename}.md"
        
        md_content = self._generate_markdown_report(data, config)
        
        with open(file_path, 'w') as f:
            f.write(md_content)
        
        return file_path
    
    def _generate_markdown_report(
        self,
        data: Dict[str, Any],
        config: ExportConfig
    ) -> str:
        """Generate Markdown report."""
        md = "# Arena Game Report\n\n"
        
        # Game summary
        md += "## Game Summary\n\n"
        md += f"- **Game ID:** {data.get('game_id', 'N/A')}\n"
        md += f"- **Date:** {data.get('start_time', 'N/A')}\n"
        md += f"- **Total Turns:** {data.get('total_turns', 0)}\n"
        md += f"- **Winner:** {data.get('winner_name', 'N/A')}\n"
        md += f"- **Total Agents:** {data.get('total_agents', 0)}\n\n"
        
        # Agents table
        if "agents" in data:
            md += "## Agent Performance\n\n"
            md += "| Agent | Final Score | Position | Status |\n"
            md += "|-------|-------------|----------|--------|\n"
            
            for agent in data["agents"]:
                name = agent.get('agent_name', agent.get('agent_id'))
                score = agent.get('final_score', 0)
                position = agent.get('final_position', 'N/A')
                status = "Winner" if agent.get('is_champion') else "Eliminated"
                md += f"| {name} | {score:.2f} | {position} | {status} |\n"
            
            md += "\n"
        
        # Key moments
        if "key_moments" in data:
            md += "## Key Moments\n\n"
            for moment in data.get("key_moments", []):
                md += f"- **Turn {moment.get('turn')}:** {moment.get('description')}\n"
            md += "\n"
        
        # Analytics
        if config.include_analytics and "analytics" in data:
            md += "## Analytics\n\n"
            analytics = data["analytics"]
            for key, value in analytics.items():
                md += f"- **{key}:** {value}\n"
            md += "\n"
        
        return md
    
    def export_batch(
        self,
        data_list: List[Dict[str, Any]],
        config: ExportConfig,
        batch_name: str = "batch"
    ) -> List[Path]:
        """
        Export multiple games/datasets.
        
        Args:
            data_list: List of data to export
            config: Export configuration
            batch_name: Batch name prefix
            
        Returns:
            List of exported file paths
        """
        exported_paths = []
        
        for i, data in enumerate(data_list):
            filename = f"{batch_name}_{i+1}"
            path = self.export_game(data, config, filename)
            exported_paths.append(path)
        
        # Create manifest
        manifest = {
            "batch_name": batch_name,
            "export_date": datetime.utcnow().isoformat(),
            "config": config.to_dict(),
            "files": [str(p) for p in exported_paths]
        }
        
        manifest_path = self.export_dir / f"{batch_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return exported_paths


class ReportGenerator:
    """
    Generates formatted reports from game and analytics data.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            template_dir: Directory containing report templates
        """
        self.template_dir = Path(template_dir) if template_dir else None
        
    def generate_game_report(
        self,
        game_data: Dict[str, Any],
        analytics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive game report.
        
        Args:
            game_data: Game data
            analytics: Optional analytics data
            
        Returns:
            Report content
        """
        report = "=" * 80 + "\n"
        report += "ARENA GAME REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Executive Summary
        report += "EXECUTIVE SUMMARY\n"
        report += "-" * 40 + "\n"
        report += f"Game ID: {game_data.get('game_id', 'N/A')}\n"
        report += f"Date: {game_data.get('start_time', 'N/A')}\n"
        report += f"Duration: {game_data.get('duration', 'N/A')}\n"
        report += f"Winner: {game_data.get('winner_name', 'N/A')}\n"
        report += f"Total Participants: {game_data.get('total_agents', 0)}\n\n"
        
        # Performance Summary
        report += "PERFORMANCE SUMMARY\n"
        report += "-" * 40 + "\n"
        
        if "agents" in game_data:
            agents = sorted(
                game_data["agents"],
                key=lambda a: a.get("final_position", 999)
            )
            
            for i, agent in enumerate(agents[:5], 1):
                report += f"{i}. {agent.get('agent_name', agent.get('agent_id'))}"
                report += f" - Score: {agent.get('final_score', 0):.2f}\n"
        
        report += "\n"
        
        # Game Progression
        report += "GAME PROGRESSION\n"
        report += "-" * 40 + "\n"
        report += f"Total Turns: {game_data.get('total_turns', 0)}\n"
        report += f"Total Messages: {game_data.get('total_messages', 0)}\n"
        report += f"Total Eliminations: {game_data.get('total_eliminations', 0)}\n"
        report += f"Avg Turn Duration: {game_data.get('avg_turn_duration', 0):.2f}s\n\n"
        
        # Analytics Section
        if analytics:
            report += "ANALYTICS\n"
            report += "-" * 40 + "\n"
            
            if "patterns" in analytics:
                report += "Detected Patterns:\n"
                for pattern, count in analytics["patterns"].items():
                    report += f"  - {pattern}: {count} occurrences\n"
                report += "\n"
            
            if "trends" in analytics:
                report += "Trends:\n"
                for metric, trend in analytics["trends"].items():
                    report += f"  - {metric}: {trend}\n"
                report += "\n"
        
        # Footer
        report += "=" * 80 + "\n"
        report += f"Report Generated: {datetime.utcnow().isoformat()}\n"
        report += "=" * 80 + "\n"
        
        return report
    
    def generate_tournament_report(
        self,
        tournament_data: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """
        Generate tournament report.
        
        Args:
            tournament_data: Tournament data
            results: Tournament results
            
        Returns:
            Report content
        """
        report = "TOURNAMENT REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Tournament ID: {tournament_data.get('tournament_id')}\n"
        report += f"Format: {tournament_data.get('format')}\n"
        report += f"Participants: {tournament_data.get('total_participants')}\n"
        report += f"Total Matches: {results.get('total_matches')}\n"
        report += f"Completed Matches: {results.get('completed_matches')}\n\n"
        
        report += "FINAL STANDINGS\n"
        report += "-" * 30 + "\n"
        
        for position, (participant, _) in enumerate(results.get('standings', [])[:10], 1):
            medal = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else ""
            report += f"{position}. {participant} {medal}\n"
        
        return report
    
    def save_report(
        self,
        report_content: str,
        filename: str,
        output_dir: str = "arena_reports"
    ) -> Path:
        """
        Save report to file.
        
        Args:
            report_content: Report content
            filename: Output filename
            output_dir: Output directory
            
        Returns:
            Path to saved report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        file_path = output_path / f"{filename}.txt"
        with open(file_path, 'w') as f:
            f.write(report_content)
        
        return file_path