# Task Completion Checklist

## MANDATORY STEPS - DO NOT SKIP!

When completing any task, follow these steps IN ORDER:

### 1. ✅ Complete Implementation
- Write all required code
- Ensure code is functional and follows best practices

### 2. ✅ Write Tests
- Create comprehensive test coverage
- Run tests and verify they pass
- Document test results

### 3. ✅ **UPDATE tasks.md IMMEDIATELY** 🚨
**THIS IS CRITICAL - DO NOT FORGET!**

```bash
# Change this:
- [ ] X.X Task name

# To this:
- [x] X.X Task name
```

**Why this matters:**
- Triggers the git hook for next task
- Tracks progress accurately
- Prevents confusion about what's done

### 4. ✅ Update git.next.hook
- Document what was completed
- List all files created/modified
- Include test results
- Specify next task clearly

### 5. ✅ Create Documentation (if needed)
- README files for major components
- Usage examples
- API documentation

### 6. ✅ Verify Hook Triggers
- Check that file changes are saved
- Confirm hook execution starts
- Verify next task begins

---

## Common Mistakes to Avoid

❌ **Forgetting to update tasks.md** - This breaks the workflow!
❌ Only updating git.next.hook without tasks.md
❌ Making formatting-only changes that don't trigger saves
❌ Updating tasks.md but not making a real content change

## Quick Reference

**Files to update on task completion:**
1. `tasks.md` - Mark task as complete `[x]`
2. `git.next.hook` - Document completion details
3. Any relevant README files

**Order matters:**
1. Code → 2. Tests → 3. tasks.md → 4. git.next.hook → 5. Docs

---

## Template for git.next.hook Updates

```markdown
## Task X.X: [Task Name] ✅

Successfully implemented [brief description]!

### Core Components (XXX+ lines):
- **Component1**: Description
- **Component2**: Description

### Features Delivered:
- Feature 1
- Feature 2

### Test Coverage (XXX+ lines):
- Test category 1
- Test category 2

## Next Steps: [Next Phase]

NEXT TASK X.X: [Next task name]

Requirements:
- Requirement 1
- Requirement 2
```

---

**REMEMBER: Update tasks.md FIRST, then everything else!**
