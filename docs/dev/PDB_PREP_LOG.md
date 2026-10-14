# PDB preparation log

Structure-specific conflicts and issues met while preparing entries with crimm: clashes,
patch choices, odd chemistry in the deposited file, results that need a non-default option.
One section per PDB ID, in alphabetical order. A general bug found this way also gets a
GitHub issue; link it here instead of repeating it.

Same writing rules as [AGENT_HANDOFF.md](AGENT_HANDOFF.md): date every entry
(`YYYY-MM-DD`), tag each claim `[ran]`, `[read]`, `[told]` or `[inferred]`, and give the crimm
commit the result was seen on.

## Entry template

```markdown
## <PDB ID>: <title of the entry from RCSB>

### YYYY-MM-DD <short name of the problem>

- seen on: crimm <commit>, <call or options used>
- symptom:
- cause:
- decision: what to do for this entry, and who decided
- evidence:
- links: issues, tests
```

## 1CRN: crambin

### 2026-10-14 Default CT3 cap clashes with the Arg10 salt bridge

- seen on: crimm `6a00566`, `TopologyGenerator().generate_model(model)` with defaults,
  then `pyCHARMMAdaptors.load_model(model)` into pyCHARMM (CHARMM 52.1 `c16e5816e`, CPU
  build). `[ran]`
- symptom: the energy before minimisation is 4.2e6 kcal/mol, nearly all van der Waals
  (VDW 4,210,926; bonds, angles and dihedrals are normal), GRMS 3.2e6. `[ran]`
- cause: `generate_model` defaults to `prot_last_patch="CT3"`
  (`crimm/Modeller/TopoLoader.py::TopologyGenerator.generate_model`). CT3 replaces the
  C-terminal carboxylate of Asn46, which the crystal has (OXT is deposited), with an
  N-methylamide. Its methyl carbon is built where the carboxylate sits in the
  Arg10–Asn46 salt bridge: Arg10 HE is 0.50 Å from Asn46 CAT and 0.85 Å from HT3. `[ran]`
- decision: CT3 stays the default; it is intended. `[told 2026-10-14]` For 1CRN, use
  `generate_model(model, prot_last_patch="CTER")`.
- evidence: with CTER, 646 atoms and total charge -1 in both crimm and CHARMM, no atom pair
  closer than 0.9 Å, total energy 178.3 kcal/mol (VDW -66.5, ELEC -418.3). With CT3, 651
  atoms and charge 0, also agreeing between crimm and CHARMM. `[ran]`
- not checked: whether other fixtures with a deposited OXT (1UBQ, 2IGD, 3PTB) show the same
  kind of clash with CT3.
- links: #107 (the planned pyCHARMM tests should load 1CRN with CTER).
