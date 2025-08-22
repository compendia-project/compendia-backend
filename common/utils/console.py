from rich.console import Console

console = Console()


def print_info(message):
    """Print info message with styling"""
    console.print(f"[bold yellow]{message}[/bold yellow]")


def print_success(message):
    """Print success message with styling"""
    console.print(f"[bold green]{message}[/bold green]")


def print_error(message):
    """Print error message with styling"""
    console.print(f"[bold red]{message}[/bold red]")
