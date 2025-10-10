# Sharding Feature Cleanup & Integration Spec

## Problem Statement

The codebase has accumulated multiple partial implementations of model sharding functionality, creating confusion about what actually works versus what was attempted. There are:

- Multiple implementations of similar functionality (ShardedModelLoader vs SelectiveModelLoader)
- Unclear which code paths are actually executed in production
- Tests that use mocks extensively, hiding real integration issues
- Documentation claiming features work that may not be fully integrated
- No clear entry point for understanding how sharding actually happens

## Goal

Audit, clean up, and properly integrate the sharding capabilities into a cohesive, working system where:

1. Any developer can trace the sharding path from API to execution
2. There's one clear implementation of each feature
3. Integration tests use real models and services
4. Documentation matches actual functionality
5. Sharding works end-to-end with real models
6. Non-sharded inference still works perfectly
7. Failures have clear, actionable error messages

## Approach

### Phase 1: Audit (Week 1)
- Map all sharding-related code
- Test each component individually with real models
- Trace actual execution paths
- Document what works vs what doesn't
- Remove confirmed dead code

### Phase 2: Consolidate & Fix (Week 2)
- Choose one model loader implementation, remove the other
- Fix inference routing to properly detect and handle sharded models
- Complete coordinator integration
- Add proper error handling

### Phase 3: Test & Document (Week 3)
- Create real end-to-end tests with small models
- Test backward compatibility
- Update documentation to match reality
- Remove aspirational documentation

### Phase 4: Validate (Week 4)
- Simplify architecture where possible
- Run all tests and verify they pass
- Measure actual memory savings and performance
- Final cleanup pass

## Key Principles

1. **Test with real models** - No more mock-only tests
2. **Audit before removing** - Understand what works before deleting
3. **Backward compatibility** - Don't break non-sharded inference
4. **Clear errors** - Failures should have actionable messages
5. **Documentation matches reality** - Remove false claims

## Files

- `requirements.md` - Detailed requirements with acceptance criteria
- `design.md` - Architecture, component design, and data flow
- `tasks.md` - Step-by-step implementation tasks (39 tasks)

## Getting Started

To execute this spec:

1. Start with Task 1.1: Create comprehensive audit document
2. Work through tasks sequentially
3. Don't skip the audit phase - it's critical
4. Test each change with real models
5. Update documentation as you go

## Success Criteria

- [ ] All tests pass (unit, integration, end-to-end)
- [ ] Sharding works with a small real model (e.g., TinyLlama)
- [ ] Non-sharded inference still works
- [ ] Memory savings are measurable and documented
- [ ] Documentation is accurate and complete
- [ ] Code is clean with no dead code or TODOs
- [ ] Any developer can understand the sharding flow

## Timeline

**Estimated Duration:** 4 weeks

- Week 1: Audit & Remove Dead Code
- Week 2: Consolidate & Fix Integration  
- Week 3: Real Testing & Documentation
- Week 4: Simplify & Validate

## Notes

- Focus on 2-worker sharding initially
- Use small models for testing (TinyLlama, Qwen-0.5B)
- Correctness over performance
- Real integration tests are mandatory
