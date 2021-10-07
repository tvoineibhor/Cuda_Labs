#pragma once
class IMemType
{
public:
	virtual ~IMemType() {}
	virtual void Inverse() = 0;
};