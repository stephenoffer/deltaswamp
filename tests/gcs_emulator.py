"""A Google Cloud Storage XML-API emulator, for the requests object_store makes.

object_store 0.13 talks to GCS through the XML API: path-style object PUT,
GET/HEAD with ranges, DELETE, `list-type=2` listings, copies, XML multipart
uploads, and the create-only precondition ``x-goog-if-generation-match: 0``
that every Delta commit's put-if-absent rests on. Neither fake-gcs-server
(1.56) nor gcp-storage-emulator implements those (fake-gcs-server takes an
XML object PUT only as a signed-URL upload and reads preconditions from the
JSON API's query string), so the store could not even create a table there.

This emulator implements exactly that surface, in-process, and like GCS it
checks the bearer token on every request: a token is valid only while its
`allow()` lifetime lasts, and an expired or unknown one gets 401. That is
what lets a test prove a refreshed token reached a running scan.

It is not GCS. It verifies what deltaswamp and object_store send and how they
react to GCS's answers (412 on a lost create, 401 on a dead token), not
GCS's own consistency or its IAM.
"""

from __future__ import annotations

import email.utils
import hashlib
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from xml.sax.saxutils import escape


@dataclass
class _Object:
    data: bytes
    generation: int
    updated: float

    @property
    def etag(self) -> str:
        return '"' + hashlib.md5(self.data, usedforsecurity=False).hexdigest() + '"'


@dataclass
class _Upload:
    bucket: str
    key: str
    parts: dict[int, bytes] = field(default_factory=dict)


class GcsEmulator:
    """Serve buckets at ``http://127.0.0.1:<port>`` until `stop()`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, dict[str, _Object]] = {}
        self._uploads: dict[str, _Upload] = {}
        self._tokens: dict[str, float | None] = {}
        self._generation = int(time.time() * 1e6)
        #: Every bearer token presented, in order (valid or not).
        self.tokens_seen: list[str] = []
        #: Requests refused for a dead or unknown token.
        self.refused = 0
        emulator = self

        class Handler(_Handler):
            owner = emulator

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def create_bucket(self, name: str) -> None:
        with self._lock:
            self._buckets.setdefault(name, {})

    def allow(self, token: str, ttl: float | None = None) -> None:
        """Accept `token` for `ttl` seconds from now (forever with None)."""
        with self._lock:
            self._tokens[token] = None if ttl is None else time.time() + ttl

    def keys(self, bucket: str) -> list[str]:
        with self._lock:
            return sorted(self._buckets.get(bucket, {}))

    # -------------------------------------------------------------- internals

    def _authorized(self, header: str | None) -> bool:
        token = (header or "").removeprefix("Bearer ").strip()
        with self._lock:
            self.tokens_seen.append(token)
            expires = self._tokens.get(token, 0.0)
            ok = token in self._tokens and (expires is None or time.time() < expires)
            if not ok:
                self.refused += 1
            return ok

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation


def _http_date(t: float) -> str:
    return email.utils.formatdate(t, usegmt=True)


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


class _Handler(BaseHTTPRequestHandler):
    owner: GcsEmulator
    protocol_version = "HTTP/1.1"
    # Headers and body are separate writes on a kept-alive connection: with
    # Nagle's algorithm on, each response waited out the client's delayed ACK.
    disable_nagle_algorithm = True

    def log_message(self, *args: Any) -> None:  # quiet
        pass

    # ------------------------------------------------------------ plumbing

    def _target(self) -> tuple[str, str, dict[str, str]]:
        parsed = urllib.parse.urlsplit(self.path)
        query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
        path = parsed.path.lstrip("/")
        bucket, _, key = path.partition("/")
        return urllib.parse.unquote(bucket), urllib.parse.unquote(key), query

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _send(
        self,
        status: int,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        *,
        head: bool = False,
    ) -> None:
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if "Content-Length" not in (headers or {}):
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body and not head:
            self.wfile.write(body)

    def _error(self, status: int, code: str) -> None:
        body = f"<?xml version='1.0' encoding='UTF-8'?><Error><Code>{code}</Code></Error>"
        self._send(status, body.encode(), {"Content-Type": "application/xml"})

    def _auth(self) -> bool:
        if self.owner._authorized(self.headers.get("Authorization")):
            return True
        self._body()
        self._error(401, "AuthenticationRequired")
        return False

    def _object_headers(self, obj: _Object) -> dict[str, str]:
        return {
            "ETag": obj.etag,
            "Last-Modified": _http_date(obj.updated),
            "x-goog-generation": str(obj.generation),
            "Content-Type": "application/octet-stream",
        }

    # -------------------------------------------------------------- verbs

    def do_GET(self) -> None:
        self._get(head=False)

    def do_HEAD(self) -> None:
        self._get(head=True)

    def _get(self, *, head: bool) -> None:
        if not self._auth():
            return
        bucket, key, query = self._target()
        owner = self.owner
        with owner._lock:
            objects = owner._buckets.get(bucket)
            if objects is None:
                return self._error(404, "NoSuchBucket")
            if not key:
                return self._list(objects, bucket, query)
            obj = objects.get(key)
        if obj is None:
            return self._error(404, "NoSuchKey")
        headers = self._object_headers(obj)
        if (match := self.headers.get("If-Match")) and match != obj.etag:
            return self._error(412, "PreconditionFailed")
        if (none := self.headers.get("If-None-Match")) and none == obj.etag:
            return self._send(304, headers=headers)
        data = obj.data
        rng = self.headers.get("Range")
        if not rng:
            return self._send(200, data, headers, head=head)
        spec = rng.removeprefix("bytes=")
        start_s, _, end_s = spec.partition("-")
        size = len(data)
        if start_s == "":
            start, end = max(0, size - int(end_s)), size - 1
        else:
            start = int(start_s)
            end = min(int(end_s), size - 1) if end_s else size - 1
        if start >= size:
            return self._error(416, "InvalidRange")
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return self._send(206, data[start : end + 1], headers, head=head)

    def _list(self, objects: dict[str, _Object], bucket: str, query: dict[str, str]) -> None:
        prefix = query.get("prefix", "")
        delimiter = query.get("delimiter")
        after = query.get("start-after", "")
        contents: list[tuple[str, _Object]] = []
        prefixes: set[str] = set()
        for key in sorted(objects):
            if not key.startswith(prefix) or (after and key <= after):
                continue
            rest = key[len(prefix) :]
            if delimiter and delimiter in rest:
                prefixes.add(prefix + rest.split(delimiter, 1)[0] + delimiter)
            else:
                contents.append((key, objects[key]))
        parts = [
            "<?xml version='1.0' encoding='UTF-8'?>",
            '<ListBucketResult xmlns="http://doc.s3.amazonaws.com/2006-03-01">',
            f"<Name>{escape(bucket)}</Name><Prefix>{escape(prefix)}</Prefix>",
            f"<KeyCount>{len(contents)}</KeyCount><IsTruncated>false</IsTruncated>",
        ]
        for key, obj in contents:
            parts.append(
                f"<Contents><Key>{escape(key)}</Key><Size>{len(obj.data)}</Size>"
                f"<LastModified>{_iso(obj.updated)}</LastModified>"
                f"<ETag>{escape(obj.etag)}</ETag></Contents>"
            )
        for p in sorted(prefixes):
            parts.append(f"<CommonPrefixes><Prefix>{escape(p)}</Prefix></CommonPrefixes>")
        parts.append("</ListBucketResult>")
        self._send(200, "".join(parts).encode(), {"Content-Type": "application/xml"})

    def do_PUT(self) -> None:
        if not self._auth():
            return
        bucket, key, query = self._target()
        body = self._body()
        owner = self.owner
        if "uploadId" in query:
            with owner._lock:
                upload = owner._uploads.get(query["uploadId"])
                if upload is None:
                    return self._error(404, "NoSuchUpload")
                upload.parts[int(query["partNumber"])] = body
            etag = '"' + hashlib.md5(body, usedforsecurity=False).hexdigest() + '"'
            return self._send(200, headers={"ETag": etag})
        source = self.headers.get("x-goog-copy-source")
        with owner._lock:
            objects = owner._buckets.get(bucket)
            if objects is None:
                return self._error(404, "NoSuchBucket")
            if source is not None:
                src_bucket, _, src_key = urllib.parse.unquote(source).lstrip("/").partition("/")
                src = owner._buckets.get(src_bucket, {}).get(src_key)
                if src is None:
                    return self._error(404, "NoSuchKey")
                body = src.data
            if not self._precondition(objects.get(key)):
                return self._error(412, "PreconditionFailed")
            obj = objects[key] = _Object(body, owner._next_generation(), time.time())
        return self._send(200, headers=self._object_headers(obj))

    def _precondition(self, existing: _Object | None) -> bool:
        want = self.headers.get("x-goog-if-generation-match")
        if want is None:
            return True
        if want == "0":
            return existing is None
        return existing is not None and str(existing.generation) == want

    def do_POST(self) -> None:
        if not self._auth():
            return
        bucket, key, query = self._target()
        body = self._body()
        owner = self.owner
        if "uploads" in query:
            upload_id = uuid.uuid4().hex
            with owner._lock:
                owner._uploads[upload_id] = _Upload(bucket, key)
            xml = (
                "<?xml version='1.0' encoding='UTF-8'?><InitiateMultipartUploadResult>"
                f"<Bucket>{escape(bucket)}</Bucket><Key>{escape(key)}</Key>"
                f"<UploadId>{upload_id}</UploadId></InitiateMultipartUploadResult>"
            )
            return self._send(200, xml.encode(), {"Content-Type": "application/xml"})
        if "uploadId" in query:
            import re

            order = [int(n) for n in re.findall(rb"<PartNumber>(\d+)</PartNumber>", body)]
            with owner._lock:
                upload = owner._uploads.pop(query["uploadId"], None)
                if upload is None:
                    return self._error(404, "NoSuchUpload")
                data = b"".join(upload.parts[n] for n in order)
                obj = owner._buckets.setdefault(bucket, {})[key] = _Object(
                    data, owner._next_generation(), time.time()
                )
            xml = (
                "<?xml version='1.0' encoding='UTF-8'?><CompleteMultipartUploadResult>"
                f"<Bucket>{escape(bucket)}</Bucket><Key>{escape(key)}</Key>"
                f"<ETag>{escape(obj.etag)}</ETag></CompleteMultipartUploadResult>"
            )
            return self._send(
                200, xml.encode(), {"Content-Type": "application/xml", **self._object_headers(obj)}
            )
        return self._error(400, "InvalidRequest")

    def do_DELETE(self) -> None:
        if not self._auth():
            return
        bucket, key, query = self._target()
        owner = self.owner
        with owner._lock:
            if "uploadId" in query:
                owner._uploads.pop(query["uploadId"], None)
                return self._send(204)
            objects = owner._buckets.get(bucket, {})
            if objects.pop(key, None) is None:
                return self._error(404, "NoSuchKey")
        return self._send(204)
