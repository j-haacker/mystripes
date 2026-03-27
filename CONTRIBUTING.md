# Contributing to MyStripes

Thanks for thinking about helping with MyStripes.

This is meant to be a low-barrier project. Small contributions are genuinely useful. A typo fix, a clearer explanation, one more test, or a small UI polish can all make the project better.

## Easy ways to help

- Report a bug or confusing behavior.
- Improve wording in the app, README, or notices.
- Add or improve tests.
- Suggest a feature or a simpler workflow.
- Fix a small bug you noticed while using the app.
- Improve the public Python API examples.

## Before you start

- It is fine to open an issue before writing code.
- It is also fine to open a small pull request directly.
- If you are unsure whether an idea fits, ask anyway.
- Please keep changes focused. Small, clear pull requests are easiest to review.

## Local setup

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests:

   ```bash
   python -m unittest discover -s tests
   ```

3. Optionally run the app locally:

   ```bash
   streamlit run app.py
   ```

If you use Pixi, the repository also includes:

```bash
pixi run test
pixi run start
```

## Coverage

To generate a local coverage report:

```bash
python -m coverage run -m unittest discover -s tests
python -m coverage report -m
```

If you want an HTML report too:

```bash
python -m coverage html
```

Then open `htmlcov/index.html` in a browser.

## Pull request tips

- Add or update tests when behavior changes.
- Keep commit messages short and descriptive.
- Mention user-visible changes in the pull request description.
- If something is incomplete, say so clearly. That is much better than hiding it.

## Community expectations

Please be kind, constructive, and patient with other contributors. We want the project to feel approachable.

## Need ideas?

Good starter contributions often include:

- clearer help text in the Streamlit app
- small layout or wording fixes
- new tests for edge cases
- README improvements
- better error messages

Thank you for helping.
