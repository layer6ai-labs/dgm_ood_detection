from typing import Optional

def buffer_loader(loader, buffer_size: Optional[int] = None, limit=None):
    if buffer_size is None:
        yield loader
    else:
    
        # tekes in a torch dataloader and returns an iterable where each
        # iteration returns a list of buffer_size batches
        for i, batch in enumerate(loader):
            if limit is not None and i // buffer_size >= limit:
                break
            if i % buffer_size == 0:
                if i != 0:
                    yield buffer
                buffer = []
            buffer.append(batch)
        if len(buffer) > 0:
            yield buffer
