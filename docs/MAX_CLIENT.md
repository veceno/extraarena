# MAX client

ExtraArena uses the same WebApp for Telegram and MAX. MAX Bridge exchanges
signed `window.WebApp.initData` for an internal game JWT through
`POST /api/auth/max`; client-decoded `initDataUnsafe` is never trusted as
authentication.

## Runtime secrets

Configure these only in the runtime secret store:

```dotenv
MAX_BOT_TOKEN=
MAX_BOT_WEBHOOK_SECRET=
MAX_BOT_USERNAME=
```

`MAX_BOT_WEBHOOK_SECRET` must be a random value of at least 32 characters.
The bot token must not be committed to the repository or passed in an URL.

## MAX platform setup

1. Set the bot Mini App URL to the production `WEBAPP_URL`.
2. Subscribe the bot to an HTTPS webhook at:
   `https://<public-host>/api/max/webhook`.
3. Send the same `MAX_BOT_WEBHOOK_SECRET` as the subscription `secret`.
4. Subscribe at least to `bot_started` and `message_created`.
5. Use `platform-api2.max.ru` and make sure the production trust store accepts
   the certificate chain required by MAX.

The webhook accepts the secret only in `X-Max-Bot-Api-Secret`. The bot exposes
`/start` and `/id`; start messages include an `open_app` button.

## Identity invariant

MAX user IDs are stored as namespaced subjects in `platform_identities` and
map to generated internal game IDs. When ExtraID is created, the immutable
`extra_account_identity_bindings` ledger records provider `max` and the
original MAX subject.

A MAX-bound ExtraID:

- cannot be deleted by the public account endpoint;
- cannot be detached or replaced by password login;
- cannot log out from a MAX launch session;
- can be linked to a new ExtraID only once;
- receives the same one-time ExtraID registration bonus as Telegram after
  email verification.

Support-assisted recovery remains the only route for attaching an existing
ExtraID to an already confirmed MAX profile.
