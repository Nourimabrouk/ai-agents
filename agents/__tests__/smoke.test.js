describe('repo smoke', () => {
  test('jest runs here', () => {
    expect(true).toBe(true);
  });

  test('package scripts include test/dev/build', () => {
    const pkg = require('../../package.json');
    expect(pkg.scripts).toBeDefined();
    expect(pkg.scripts.test).toBeDefined();
    expect(pkg.scripts.dev).toBeDefined();
    expect(pkg.scripts.build).toBeDefined();
  });
});

