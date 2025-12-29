# Contributing to MOUAADNET-ULTRA

Thank you for your interest in contributing to MOUAADNET-ULTRA! 🎉

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/mouaadnet-ultra.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make changes and commit: `git commit -m "feat: add new feature"`
5. Push and create a Pull Request

## 📋 Development Setup

```bash
# Clone repository
git clone https://github.com/mouaadidoufkir/mouaadnet-ultra.git
cd mouaadnet-ultra

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 📝 Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black mouaadnet_ultra/
isort mouaadnet_ultra/

# Check linting
flake8 mouaadnet_ultra/
mypy mouaadnet_ultra/
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=mouaadnet_ultra --cov-report=html

# Run specific test
pytest tests/test_backbone.py -v
```

## 📚 Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Examples:
```
feat: add spatial attention to gender head
fix: correct stride calculation in PConv
docs: update architecture diagram
```

## 🔀 Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

## 🐛 Reporting Issues

Please include:
- Python version
- PyTorch version
- Operating system
- Minimal reproducible example
- Full error traceback

## 💡 Feature Requests

We love new ideas! Please:
- Check existing issues first
- Describe the use case
- Explain expected behavior
- Consider implementation approach

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Questions? Open an issue or reach out to the maintainers!
