# HawkEye — Ethics, Privacy & Legal (India)

Face recognition + automated dispatch is sensitive. HawkEye is deliberately
**responsible-by-design** — and this is a project strength, not a disclaimer.

## What we do

- **Local, consented identity set only.** ~20 enrolled people who have given
  consent. **No Aadhaar / government-ID integration.** There is no public Aadhaar
  face-matching API, and doing so would be legally fraught under the *Digital
  Personal Data Protection Act, 2023 (DPDP)* and the *Puttaswamy* privacy
  judgment. Real government-ID integration is explicitly **future work, pending
  official API access and legal review** — not part of any current build.
- **Explainable, gated actions.** Every automated dispatch is severity- and
  confidence-gated, cooldown-protected, and written to an auditable action log.
- **Human-in-the-loop.** The dashboard surfaces every incident with its evidence
  so a human can confirm/dismiss; the two-stage cascade exists partly to *avoid*
  false dispatches.
- **Evidence minimization.** Faces/clips are saved only for confirmed incidents.

## Phase 3 commitments (production)

- Encrypt the biometric DB and evidence at rest; per-user access control on the
  dashboard.
- Consent records + configurable retention + audit logs aligned to DPDP 2023
  (purpose limitation, data minimization, storage limitation).
- Clear signage / notice requirements for any monitored space.
- Real dispatch integrations stay behind explicit configuration and prominent
  disclaimers; never enabled by default.

## Known limitations to state honestly (thesis)

- Face recognition has demographic bias; a local 20-person set is a demo, not a
  deployed surveillance claim.
- Anomaly detection can err; automated response must remain assistive, with human
  oversight, never fully autonomous enforcement.
- Drone flight over public spaces is regulated by **DGCA**; Phase 2 autonomy is
  developed in simulation and demonstrated indoors.
